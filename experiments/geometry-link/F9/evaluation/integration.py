"""Six fixed native engineering frames; full-state/receive-only/DDP gates, no AP."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np
import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(HERE), str(ROOT / 'src'), str(ROOT / 'experiments/geometry-link/F9/code')]
from common import (check, sha256, source_identity, frozen_load, treatment, F9Observer,
                    validate_native_record, install_native3D_forward)
from conditions import PARENT_SHA, array_evidence
from geocomm.inference import SENSOR_INPUT_KEYS
from geocomm.pooling_diagnostic import state_hashes
from receiver_reference import ReceiverReference, prediction_arrays, value_identity

IDS = ['000036', '000054', '000071', '000082', '000113', '000141']
SPLIT_SHA = '5cc02c423db12bfe3f5d5a8e7ddbe7c307b32d2c1e77cf8bd5a8932f0f9b9ba1'


def source_specs():
    liga = ROOT / 'third_party/LIGA-Stereo'
    return dict(project=(ROOT, ['src', 'scripts', 'configs', 'pyproject.toml']),
                liga=(liga, ['liga', 'configs', 'tools', 'setup.py']),
                mmdet=(ROOT / 'third_party/mmdetection_kitti', ['mmdet']),
                stereo_rcnn=(ROOT / 'third_party/Stereo-RCNN', ['lib', 'demo.py', 'test_net.py']),
                F7=(ROOT, ['experiments/geometry-link/F7/code']),
                F8=(ROOT, ['experiments/geometry-link/F8/code']),
                experiment=(ROOT, ['experiments/geometry-link/F9/code']),
                evaluation=(ROOT, ['experiments/geometry-link/F9/evaluation']))


def sensor_identity(batch):
    check(set(batch) == set(SENSOR_INPUT_KEYS), 'complete sensor-only evaluation input')
    return dict(frame_id=str(batch['frame_id'][0]),
                arrays={k: array_evidence(batch[k]) for k in ('left_img', 'right_img', 'image_shape')},
                calibration=value_identity(batch['calib']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('U', 'G', 'P', 'S'), required=True)
    parser.add_argument('--channel', choices=('identity', 'awgn'), required=True)
    for name in ('output-dir', 'report', 'source-manifest'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve(); args.report = args.report.resolve()
    check(not args.output_dir.exists() and not args.report.exists(), 'unique smoke outputs')
    args.output_dir.mkdir(parents=True)
    config_path = HERE / 'configs' / (args.channel + '10-holdout.yaml')
    sources = {k: source_identity(*v) for k, v in source_specs().items()}
    expected_sources = json.loads(args.source_manifest.read_text())
    check(sources == {k: v['actual_sources'] for k, v in expected_sources.items()}, 'full eight-tree source freeze')
    closure_path = ROOT / 'data/provenance/stereo-epipolar-native-sanity-001-closure.json'
    closure = json.loads(closure_path.read_text())
    initial = Path(f'/mnt/d/paper6/runs/stereo-epipolar-native-sanity-001-{args.arm}/initialization.pth')
    expected_sha = closure['native_large_sha256'][args.arm][str(initial)]
    check(closure['actual_terminal'] and closure['actual_training_records'] == 24, 'complete prior engineering closure')
    parent = Path('/mnt/d/paper6/runs/stereo-encoder-native-seed17-001-joint/checkpoint_epoch_1.pth')
    check(sha256(parent) == PARENT_SHA and sha256(initial) == expected_sha, 'fixed zero-update snapshot and parent')
    memory = subprocess_memory()
    check(memory['free_bytes'] >= 12 * 2**30, 'fixed physical GPU margin')
    info = dict(state='starting', engineering_only=True, no_AP=True, arm=args.arm, channel=args.channel,
                snr_db=10., ids=IDS, frames=0, seed=17, started_unix=time.time(),
                source_identities=sources, source_manifest_sha256=sha256(args.source_manifest),
                config_sha256=sha256(config_path), initialization_sha256=expected_sha,
                engineering_closure_sha256=sha256(closure_path), sole_parent_sha256=PARENT_SHA,
                output_dir=str(args.output_dir), gpu_margin_before=memory, torch=torch.__version__)
    def save(): args.report.write_text(json.dumps(info, indent=2, allow_nan=False) + '\n')
    save(); observer = reference = None; group = None
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
        random.seed(17); np.random.seed(17); torch.manual_seed(17); torch.cuda.manual_seed_all(17)
        torch.backends.cudnn.benchmark = False
        liga = ROOT / 'third_party/LIGA-Stereo'
        sys.path[:0] = [str(liga), str(ROOT / 'third_party/mmdetection_kitti')]
        os.chdir(liga)
        from easydict import EasyDict
        from liga.config import cfg_from_yaml_file
        from liga.datasets import build_dataloader
        from liga.models import build_network, load_data_to_gpu
        from liga.utils.common_utils import create_logger
        config = cfg_from_yaml_file(str(config_path), EasyDict())
        dataset, loader, _ = build_dataloader(config.DATA_CONFIG, config.CLASS_NAMES,
                batch_size=1, dist=False, workers=4, training=False, logger=create_logger(args.output_dir / 'dataset.log'))
        split = dataset.root_path / 'ImageSets' / (dataset.split + '.txt')
        check(dataset.split == 'geocomm_tune_holdout' and len(dataset) == 372 and
              split.read_text().split()[:6] == IDS and sha256(split) == SPLIT_SHA, 'locked ordered internal six IDs')
        check(json.loads((dataset.root_path / 'download-manifest.json').read_text())['state'] == 'finished', 'verified KITTI archives')
        info['dataset'] = dict(root=str(dataset.root_path.resolve()), split_sha256=sha256(split),
            infos={str(p): sha256(dataset.root_path / p) for p in dataset.dataset_cfg.INFO_PATH[dataset.mode]})
        group = tempfile.TemporaryDirectory(prefix='F9-inference-group-')
        dist.init_process_group('nccl', init_method='file://' + group.name + '/rank', rank=0, world_size=1)
        model = build_network(config.MODEL, len(config.CLASS_NAMES), dataset).eval()
        treatment(model, args.arm, args.channel, 10.)
        loading = frozen_load(model, initial, expected_sha, role='engineering_initialization')
        old = torch.load(parent, map_location='cpu', weights_only=False)['model_state']
        check(len(old) == 535 and all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in old.items()), 'all535 parent values unchanged')
        head = model.backbone_3d.stereo_feature_link.gain_head
        check(torch.count_nonzero(head[2].weight) == torch.count_nonzero(head[2].bias) == 0, 'zero output head')
        del old
        model = model.cuda().eval(); install_native3D_forward(model)
        before = state_hashes(model)
        check(before == loading['state_hashes'], '539 exact CPU-to-GPU state preservation')
        rows = []
        with (args.output_dir / 'records.jsonl').open('x') as stream:
            def record(row):
                validate_native_record(row, args.arm, args.channel, 10.)
                stream.write(json.dumps(row, allow_nan=False) + '\n'); stream.flush(); rows.append(row)
            observer = F9Observer(model, args.arm, args.channel, 10., record)
            reference = ReceiverReference(model, observer)
            wrapped = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0], broadcast_buffers=False)
            wrapped.eval()
            inputs = []
            for index, batch in enumerate(loader):
                sensors = {k: batch[k] for k in SENSOR_INPUT_KEYS}
                load_data_to_gpu(sensors)
                fingerprint = sensor_identity(sensors)
                check(fingerprint['frame_id'] == IDS[index], 'exact smoke order')
                with torch.no_grad():
                    predictions, diagnostics = wrapped(sensors)
                    check(not diagnostics, 'no post-prediction label/recall/AP calculation in smoke')
                    arrays = prediction_arrays(predictions)
                    path = args.output_dir / (IDS[index] + '-predictions.npz')
                    with path.open('xb') as output: np.savez_compressed(output, **arrays)
                    if index == 0:
                        reference_report, paired_arrays = reference.compare(predictions)
                        path_ref = args.output_dir / 'receiver-reference.npz'
                        with path_ref.open('xb') as output: np.savez_compressed(output, **paired_arrays)
                        info['reference'] = dict(**reference_report, raw_arrays_sha256=sha256(path_ref))
                check(sensor_identity(sensors) == fingerprint, 'native inference mutated input sensors')
                inputs.append(fingerprint); info['frames'] = index + 1; save()
                del predictions, diagnostics, arrays, sensors, batch
                if index == 5: break
        check(len(rows) == 6 and [r['frame_id'] for r in rows] == IDS, 'exact48-frame integration budget')
        check(observer.calls == dict(frames=6, student=12, codec=6, channel=6, build_cost=6, forbidden=0), 'exact native chronology')
        check(all(r['F9_coding']['amplitude_gains'] == [1.] * 64 for r in rows), 'zero-head initialization gains')
        check(state_hashes(model) == before, 'all539 inference states read-only')
        check(sources == {k: source_identity(*v) for k, v in source_specs().items()}, 'eight-tree sources changed')
        check(sha256(initial) == expected_sha and sha256(parent) == PARENT_SHA, 'snapshot/parent changed')
        info.update(state='passed', calls=observer.calls, reference_calls=observer.reference_calls,
                    checkpoint_loading=loading, before_state_hashes=before, after_state_hashes=state_hashes(model),
                    parent_states_exact=535, receiver_execution=model.F9_receiver_execution, sensor_identities=inputs,
                    raw_artifacts_sha256={p.name: sha256(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()},
                    peak_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30)
    except BaseException as error:
        info.update(state='failed', exception=repr(error)); raise
    finally:
        if reference is not None: reference.close()
        if observer is not None: observer.close()
        if dist.is_initialized(): dist.destroy_process_group()
        if group is not None: group.cleanup()
        info['ended_unix'] = time.time(); save()


def subprocess_memory():
    import subprocess
    values = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                                     '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    check(len(values) == 1, 'single NVIDIA GPU')
    used, total = (int(x.strip()) for x in values[0].split(','))
    return dict(used_bytes=used * 2**20, total_bytes=total * 2**20, free_bytes=(total-used) * 2**20)


if __name__ == '__main__': main()
