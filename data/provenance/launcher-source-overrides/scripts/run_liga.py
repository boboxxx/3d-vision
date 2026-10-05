#!/usr/bin/env python3
"""Launch full upstream detector with explicit seeds and initialization.

    python scripts/run_liga.py train --seed 17 --init-ckpt FILE --codec-only \
      --cfg_file CONFIG --launcher pytorch

Remaining options are passed to the upstream CLI. Does not change the dataset,
resolution, evaluator or training epoch budget.
"""
import argparse
import json
import os
from pathlib import Path
import random
import runpy
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from geocomm.evidence import revision, sha256, source_identity, serializable


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "test"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--liga-root", type=Path, default=ROOT / "third_party/LIGA-Stereo")
    p.add_argument("--init-ckpt", type=Path)
    p.add_argument("--codec-only", action="store_true")
    p.add_argument("--manifest", type=Path, required=True)
    args, remainder = p.parse_known_args()
    if "--fix_random_seed" in remainder:
        p.error("use --seed; upstream --fix_random_seed overwrites it")
    if "--cfg_file" not in remainder:
        p.error("--cfg_file is required")
    if args.codec_only and args.mode != "train":
        p.error("--codec-only is a training stage")
    cfg_path = Path(remainder[remainder.index("--cfg_file") + 1]).resolve()
    remainder[remainder.index("--cfg_file") + 1] = str(cfg_path)
    run_ckpt = None
    if "--ckpt" in remainder:
        checkpoint_index = remainder.index("--ckpt") + 1
        run_ckpt = Path(remainder[checkpoint_index]).resolve()
        remainder[checkpoint_index] = str(run_ckpt)
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    seed = args.seed + rank
    os.environ["GEOCOMM_RANK_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    sys.path[:0] = [str(args.liga_root.resolve()), str(args.liga_root.resolve() / "tools")]
    import liga.models
    original_builder = liga.models.build_network
    init_ckpt = args.init_ckpt.resolve() if args.init_ckpt else None
    manifest = args.manifest.resolve()
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        manifest = manifest.with_name(manifest.stem + ".rank%d" % rank + manifest.suffix)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest.exists():
        p.error("manifest already exists; choose a unique run id to preserve evidence")
    info = {
        "mode": args.mode, "seed": args.seed, "rank_seed": seed,
        "code_revision": revision(ROOT), "upstream_revision": revision(args.liga_root),
        "project_sources": source_identity(ROOT, ["src", "scripts", "configs", "pyproject.toml"]),
        "detector_sources": source_identity(args.liga_root, ["liga", "configs", "tools", "setup.py"]),
        "mmdet_sources": source_identity(ROOT / "third_party/mmdetection_kitti", ["mmdet"]),
        "config_sha256": sha256(cfg_path), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "initialization_sha256": sha256(init_ckpt) if init_ckpt else None,
        "run_checkpoint_sha256": sha256(run_ckpt) if run_ckpt else None,
        "codec_only": args.codec_only, "arguments": remainder,
        "started_at_unix": time.time(), "state": "starting",
    }
    manifest.write_text(json.dumps(info, indent=2), encoding="utf-8")

    def build_with_initialization(*pargs, **kwargs):
        model = original_builder(*pargs, **kwargs)
        if args.mode == "test":
            dataset = kwargs.get('dataset')
            if dataset is None:
                raise RuntimeError('benchmark dataset must be explicit')
            data_root = dataset.root_path
            download = data_root / 'download-manifest.json'
            if json.loads(download.read_text())['state'] != 'finished':
                raise RuntimeError('benchmark requires completed verified KITTI archives')
            split = data_root / 'ImageSets' / (dataset.split + '.txt')
            ids = split.read_text().split()
            if len(ids) != len(dataset) or len(set(ids)) != len(ids):
                raise RuntimeError('benchmark dataset count differs from split')
            info['dataset'] = dict(root=str(data_root), count=len(dataset),
                split_sha256=sha256(split), download_manifest_sha256=sha256(download),
                infos={str(path): sha256(data_root / path)
                       for path in dataset.dataset_cfg.INFO_PATH[dataset.mode]})
            from geocomm.inference import sensor_only_prediction, SENSOR_INPUT_KEYS
            upstream_forward = model.forward
            def sensor_forward(batch):
                if model.training:
                    raise RuntimeError('sensor-only evaluation wrapper called in training mode')
                if 'test_inference_inputs' not in info:
                    info['test_inference_inputs'] = [key for key in SENSOR_INPUT_KEYS if key in batch]
                    info['labels_and_LiDAR_access'] = 'evaluator_only_after_predictions'
                    manifest.write_text(json.dumps(info, indent=2), encoding='utf-8')
                return sensor_only_prediction(upstream_forward, batch, model.generate_recall_record,
                    model.model_cfg.POST_PROCESSING.RECALL_THRESH_LIST)
            model.forward = sensor_forward
            upstream_loader = model.load_params_from_file
            def checked_loader(filename, **load_kwargs):
                expected = {k: v.shape for k, v in model.state_dict().items()}
                if expected:
                    # Upstream permits partial checkpoints. A clean detector
                    # checkpoint must never silently test a random codec.
                    checkpoint = torch.load(filename, map_location="cpu")
                    from geocomm.compat import adapt_spconv_state
                    state = adapt_spconv_state(model, checkpoint["model_state"])
                    missing = [k for k, shape in expected.items()
                               if k not in state or state[k].shape != shape]
                    del checkpoint, state
                    if missing:
                        raise RuntimeError("Evaluation checkpoint lacks required model weights: " + str(missing))
                return upstream_loader(filename=filename, **load_kwargs)
            model.load_params_from_file = checked_loader
        if init_ckpt:
            from liga.utils.common_utils import create_logger
            model.load_params_from_file(str(init_ckpt), to_cpu=True, logger=create_logger())
        if args.codec_only:
            for name, param in model.named_parameters():
                param.requires_grad_("semantic_link" in name)
            frozen_bn = [m for name, m in model.named_modules()
                         if "semantic_link" not in name and isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
            def freeze_stats(module, inputs):
                for bn in frozen_bn:
                    bn.eval()
            model.register_forward_pre_hook(freeze_stats)
        # Unused generic scorer must not cause DDP hangs in geometry/uniform runs.
        for m in model.modules():
            if hasattr(m, "generic_score") and getattr(m, "allocation", None) != "learned":
                for param in m.generic_score.parameters():
                    param.requires_grad_(False)
        info["trainable_parameter_names"] = [name for name, param in model.named_parameters() if param.requires_grad]
        info["state"] = "model_built"
        manifest.write_text(json.dumps(info, indent=2), encoding="utf-8")
        return model

    liga.models.build_network = build_with_initialization
    if args.mode == 'test':
        from eval_utils import eval_utils
        full_evaluation = eval_utils.eval_one_epoch
        def evaluation_with_metrics(*pargs, **kwargs):
            result_dir = kwargs['result_dir']
            if any((result_dir / name).exists() for name in ('metrics.json', 'result.pkl', 'metric_result.pkl')):
                raise RuntimeError('retain earlier evaluation artifacts; choose unique eval_tag')
            result = full_evaluation(*pargs, **kwargs)
            if rank == 0:
                metrics_path = result_dir / 'metrics.json'
                metrics_path.write_text(json.dumps(result, indent=2, default=serializable,
                    allow_nan=False) + '\n', encoding='utf-8')
                info['metrics'] = dict(path=str(metrics_path), sha256=sha256(metrics_path))
            return result
        eval_utils.eval_one_epoch = evaluation_with_metrics
    os.chdir(str(args.liga_root.resolve()))
    sys.argv = ["tools/" + args.mode + ".py"] + remainder
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
        info["state"] = "finished"
    except BaseException as exc:
        info["state"] = "failed"
        info["exception"] = repr(exc)
        raise
    finally:
        info["ended_at_unix"] = time.time()
        manifest.write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
