#!/usr/bin/env python3
"""F1 fixed-budget student feature pretraining on the native training-only fold.

The complete detector is initialized and retained. Only the student is executed
with gradients; neither the codec nor the original detector training forward runs.
"""
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

ROOT = Path(__file__).resolve().parents[1]
LIGA = ROOT / 'third_party/LIGA-Stereo'
sys.path[:0] = [str(ROOT / 'src'), str(LIGA), str(ROOT / 'third_party/mmdetection_kitti')]
from geocomm.compat import adapt_spconv_state
from geocomm.evidence import revision, sha256, source_identity, serializable
from geocomm.student import teacher_feature_targets, feature_distillation
from geocomm.feature_agreement import feature_agreement, assert_original_state, INTERFACES


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=serializable))
    temporary.replace(path)


def fold_identity(dataset, fold):
    expected = fold['folds'][dataset.split]
    root = dataset.root_path.resolve()
    split = root / 'ImageSets' / (dataset.split + '.txt')
    paths = dataset.dataset_cfg.INFO_PATH[dataset.mode]
    if len(paths) != 1 or split.read_text().split() != expected['ids']:
        raise RuntimeError('native dataset split differs from locked tuning fold')
    infos = root / paths[0]
    if (len(dataset) != expected['count'] or sha256(split) != expected['split_sha256']
            or sha256(infos) != expected['infos_sha256']):
        raise RuntimeError('native dataset identity differs from locked tuning fold')
    download = root / 'download-manifest.json'
    if json.loads(download.read_text())['state'] != 'finished':
        raise RuntimeError('verified KITTI preparation required')
    return dict(root=str(root), split=dataset.split, count=len(dataset),
        split_sha256=sha256(split), infos={str(paths[0]): sha256(infos)},
        download_manifest_sha256=sha256(download))


def feature_forward(backbone, batch):
    # Native loader retains labels/LiDAR for augmentation, but feature functions
    # receive only the two normalized and jointly augmented RGB images.
    images = [torch.as_tensor(batch[key], device='cuda', dtype=torch.float32)
              for key in ('left_img', 'right_img')]
    targets = teacher_feature_targets(backbone.feature_backbone, backbone.feature_neck, *images)
    left_stereo, left_appearance = backbone.student_semantic_link_encoder(images[0])
    right_stereo, _ = backbone.student_semantic_link_encoder(images[1])
    students = (left_stereo, right_stereo, left_appearance)
    return feature_distillation(students, targets), feature_agreement(students, targets)


def evaluate_features(backbone, loader, output, engineering_steps):
    # Evaluation does not change the following epoch's training RNG sequence.
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
    student = backbone.student_semantic_link_encoder
    prior_mode = student.training
    totals, ids = {}, []
    try:
        student.eval()
        with output.open('x') as stream, torch.no_grad():
            for index, batch in enumerate(loader):
                loss, agreement = feature_forward(backbone, batch)
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite holdout loss')
                frame = str(batch['frame_id'][0])
                ids.append(frame)
                stream.write(json.dumps(dict(frame_id=frame, loss=float(loss), interfaces=agreement),
                                        allow_nan=False) + '\n')
                for name in INTERFACES:
                    for key, value in agreement[name].items():
                        totals[(name, key)] = totals.get((name, key), 0.) + value
                if engineering_steps and index + 1 >= engineering_steps:
                    break
        expected = loader.dataset.sample_id_list
        if not engineering_steps and ids != expected:
            raise RuntimeError('feature holdout coverage/order differs')
        return dict(frames=len(ids), records_sha256=sha256(output),
            aggregation='arithmetic mean of per-frame interface metrics',
            interfaces={name: {key: totals[(name, key)]/len(ids)
                        for key in ('student_rms', 'teacher_rms', 'normalized_mse', 'cosine')}
                        for name in INTERFACES})
    finally:
        student.train(prior_mode)
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        torch.cuda.set_rng_state_all(rng[3])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--fold', type=Path, default=ROOT/'data/internal-tuning-fold-001.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--engineering-steps', type=int, default=0,
                        help='separate partial real-data sanity run, never F1 experimental evidence')
    args = parser.parse_args()
    for key in ('config', 'checkpoint', 'protocol', 'fold', 'output_dir', 'manifest'):
        setattr(args, key, getattr(args, key).resolve())
    if args.output_dir.exists() or args.manifest.exists() or args.engineering_steps < 0:
        parser.error('preserve earlier evidence; unique output directory/manifest required')
    args.output_dir.mkdir(parents=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    info = dict(evidence_type='engineering_only' if args.engineering_steps else 'exploratory_feature_pretraining',
        state='starting', started_at_unix=time.time(), seed=17,
        protocol_sha256=sha256(args.protocol), config_sha256=sha256(args.config),
        initialization_sha256=sha256(args.checkpoint), fold_sha256=sha256(args.fold),
        code_revision=revision(ROOT), output_dir=str(args.output_dir), torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0), cuda_runtime=torch.version.cuda,
        project_sources=source_identity(ROOT, ['src', 'scripts', 'configs', 'pyproject.toml']),
        detector_sources=source_identity(LIGA, ['liga', 'configs', 'tools', 'setup.py']),
        mmdet_sources=source_identity(ROOT/'third_party/mmdetection_kitti', ['mmdet']),
        optimizer_steps=0, completed_epochs=0, epochs=[])
    write_json(args.manifest, info)
    process_group_directory = None
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        torch.cuda.manual_seed_all(17)
        os.chdir(LIGA)
        from easydict import EasyDict
        from liga.config import cfg_from_yaml_file
        from liga.datasets import build_dataloader
        from liga.models import build_network
        from liga.utils.common_utils import create_logger
        config = cfg_from_yaml_file(str(args.config), EasyDict())
        opt = config.OPTIMIZATION
        if (opt.NUM_EPOCHS, opt.BATCH_SIZE_PER_GPU, opt.LR, opt.WEIGHT_DECAY, opt.GRAD_NORM_CLIP) != (5, 1, .001, .0001, 10):
            raise RuntimeError('F1 fixed optimization protocol differs')
        if (config.MODEL.BACKBONE_3D.SEMANTIC_LINK.enabled
                or not config.MODEL.BACKBONE_3D.STUDENT_ENCODER.enabled):
            raise RuntimeError('F1 requires student and disabled communication')
        logger = create_logger(args.output_dir/'native_dataset.log')
        train_set, train_loader, _ = build_dataloader(config.DATA_CONFIG, config.CLASS_NAMES,
            batch_size=1, dist=False, workers=4, training=True, logger=logger)
        holdout_set, holdout_loader, _ = build_dataloader(config.DATA_CONFIG, config.CLASS_NAMES,
            batch_size=1, dist=False, workers=4, training=False, logger=logger)
        fold = json.loads(args.fold.read_text())
        info['dataset'] = fold_identity(train_set, fold)
        info['holdout_dataset'] = fold_identity(holdout_set, fold)
        if len(train_loader) != 3340 or len(holdout_loader) != 372:
            raise RuntimeError('F1 locked sample/update budget differs')
        # The original LiDAR teacher constructor asks for dist.get_rank even in
        # a one-GPU build. Use an isolated single-rank group, without DDP.
        process_group_directory = tempfile.TemporaryDirectory(prefix='F1-process-group-')
        dist.init_process_group('gloo', init_method='file://' + process_group_directory.name + '/rank',
                                rank=0, world_size=1)
        model = build_network(config.MODEL, len(config.CLASS_NAMES), train_set).cuda().eval()
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        reference = adapt_spconv_state(model, checkpoint['model_state'])
        expected = model.state_dict()
        missing = [name for name in expected if name not in reference
                   and not name.startswith('backbone_3d.student_semantic_link_encoder.')]
        wrong = [name for name, value in reference.items()
                 if name not in expected or value.shape != expected[name].shape]
        if len(reference) != 484 or missing or wrong:
            raise RuntimeError('incomplete author initialization: ' + repr((missing, wrong)))
        model.load_state_dict(dict(expected, **reference), strict=True)
        # CPU copies remain independent of model storage throughout all epochs.
        reference = {name: value.detach().cpu().clone() for name, value in reference.items()}
        info['initialization_epoch'] = checkpoint.get('epoch')
        del checkpoint, expected
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        student = model.backbone_3d.student_semantic_link_encoder
        for parameter in student.parameters():
            parameter.requires_grad_(True)
        parameters = list(student.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=.0001)
        info.update(state='training', trainable_parameter_names=[name for name, p in model.named_parameters() if p.requires_grad],
            trainable_parameters=sum(p.numel() for p in parameters), original_tensors_checked=assert_original_state(model, reference),
            budget=dict(epochs=5, train_frames=3340, updates=16700, holdout_frames=372,
                        batch_size=1, workers=4, lr=.001, weight_decay=.0001, clip_norm=10),
            initialization_epoch=info['initialization_epoch'], training_inputs=['left_img', 'right_img'],
            teacher_exposures_per_training_frame=2, inference_teacher_allowed=False,
            holdout_limitation=fold['limitation'])
        write_json(args.manifest, info)
        scalars = args.output_dir/'training.jsonl'
        steps = 0
        with scalars.open('x') as stream:
            for epoch in range(1, 6):
                student.train()
                epoch_ids = []
                for batch in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    loss, agreement = feature_forward(model.backbone_3d, batch)
                    if not torch.isfinite(loss):
                        raise RuntimeError('nonfinite loss before optimizer update')
                    loss.backward()
                    if any(p.grad is None for p in parameters):
                        raise RuntimeError('missing student gradient before optimizer update')
                    norm = torch.nn.utils.clip_grad_norm_(parameters, 10., error_if_nonfinite=True)
                    optimizer.step()
                    steps += 1
                    info['optimizer_steps'] = steps
                    frame = str(batch['frame_id'][0])
                    epoch_ids.append(frame)
                    stream.write(json.dumps(dict(step=steps, epoch=epoch, frame_id=frame,
                        loss=float(loss.detach()), preclip_grad_norm=float(norm), interfaces=agreement), allow_nan=False) + '\n')
                    if steps % 100 == 0 or steps == 1:
                        stream.flush()
                        write_json(args.manifest, info)
                        print(f'epoch={epoch} step={steps} loss={float(loss.detach()):.6f} grad={float(norm):.6f}', flush=True)
                    if args.engineering_steps and steps >= args.engineering_steps:
                        break
                # Native LIGA replaces examples whose augmented GT becomes empty.
                # Keep that behavior and record actual IDs, rather than falsely
                # asserting each returned frame occurs exactly once per epoch.
                complete = len(epoch_ids) == 3340 and set(epoch_ids) <= set(train_set.sample_id_list)
                if not complete and not args.engineering_steps:
                    raise RuntimeError('training epoch count or frame membership differs')
                checked = assert_original_state(model, reference)
                snapshot = args.output_dir/f'checkpoint_epoch_{epoch if complete else 0}.pth'
                # Whole model format compatible with strict full-detector testing.
                torch.save(dict(model_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    optimizer_state=optimizer.state_dict(), epoch=epoch if complete else 0,
                    it=steps, version='F1-student-only', rng_state=dict(python=random.getstate(),
                    numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())), snapshot)
                holdout = evaluate_features(model.backbone_3d, holdout_loader,
                    args.output_dir/f'feature_holdout_epoch_{epoch}.jsonl', args.engineering_steps)
                assert_original_state(model, reference)
                epoch_result = dict(epoch=epoch, complete_training_epoch=complete, updates=len(epoch_ids),
                    unique_returned_training_frames=len(set(epoch_ids)),
                    native_empty_augmented_GT_resampling_retained=True,
                    checkpoint_path=str(snapshot), checkpoint_sha256=sha256(snapshot), original_states_identical=checked,
                    holdout=holdout, completed_at_unix=time.time())
                info['epochs'].append(epoch_result)
                info['completed_epochs'] += int(complete)
                write_json(args.output_dir/f'epoch_{epoch}.json', epoch_result)
                stream.flush()
                write_json(args.manifest, info)
                print(json.dumps(epoch_result, allow_nan=False), flush=True)
                if args.engineering_steps:
                    break
        if not args.engineering_steps and (steps != 16700 or info['completed_epochs'] != 5):
            raise RuntimeError('incomplete F1 update budget')
        info.update(state='finished', training_records_sha256=sha256(scalars), ended_at_unix=time.time(),
                    original_states_identical=assert_original_state(model, reference))
        write_json(args.manifest, info)
    except BaseException as exc:
        info.update(state='failed', exception=repr(exc), ended_at_unix=time.time())
        write_json(args.manifest, info)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if process_group_directory is not None:
            process_group_directory.cleanup()


if __name__ == '__main__':
    main()
