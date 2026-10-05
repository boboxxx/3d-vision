#!/usr/bin/env python3
"""Audit codec-only F0 checkpoint against every original author state tensor."""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from geocomm.evidence import sha256

NORMALIZATION_BUFFERS = {
    'dense_head.norm_imitation.spatial_features_2d.scale',
    'dense_head.norm_imitation.volume_features.scale',
}


def frozen_state(reference, trained, steps):
    converted, updated_buffers, unchanged = [], [], []
    for key, original in reference.items():
        if key not in trained:
            raise ValueError('original tensor missing: '+key)
        actual = trained[key]
        if not torch.isfinite(actual).all():
            raise ValueError('nonfinite original tensor: '+key)
        if original.shape != actual.shape:
            # Identified released LiDAR backbone RSCK -> KRSC kernels only.
            if not (key.startswith('lidar_model.backbone_3d.') and
                    key.endswith('.weight') and original.ndim == 5):
                raise ValueError('unidentified layout change: '+key)
            original = original.permute(4, 0, 1, 2, 3).contiguous()
            if original.shape != actual.shape:
                raise ValueError('sparse layout mismatch: '+key)
            converted.append(key)
        if key == 'global_step':
            if not torch.equal(actual, original + steps):
                raise ValueError('global step differs from locked update count')
            updated_buffers.append(key)
        elif key in NORMALIZATION_BUFFERS:
            if original.dtype != actual.dtype or original.shape != actual.shape or not (actual > 0).all():
                raise ValueError('invalid training-only normalization buffer')
            updated_buffers.append(key)
        elif not torch.equal(actual, original):
            raise ValueError('frozen original tensor changed: '+key)
        else:
            unchanged.append(key)
    return dict(unchanged_original_tensors=len(unchanged),
                allowed_training_only_buffers=updated_buffers,
                value_preserving_sparse_layout_keys=converted)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--initialization', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--expected-steps', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('retain previous audit; choose unique output')
    run = json.loads(args.manifest.read_text())
    if run['mode'] != 'train' or not run['codec_only']:
        raise ValueError('not a codec-only run')
    if sha256(args.initialization) != run['initialization_sha256']:
        raise ValueError('initialization identity mismatch')
    initial = torch.load(args.initialization, map_location='cpu', weights_only=False)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if len(initial['model_state']) != run['matched_initialization_tensors']:
        raise ValueError('initialization tensor count mismatch')
    if checkpoint['epoch'] != 1 or checkpoint['it'] != args.expected_steps:
        raise ValueError('checkpoint does not match F0 epoch/update budget')
    report = frozen_state(initial['model_state'], checkpoint['model_state'], args.expected_steps)
    names = run['trainable_parameter_names']
    if not names or any('semantic_link' not in name for name in names):
        raise ValueError('unexpected trainable receiver parameter')
    optimizer = checkpoint['optimizer_state']
    ids = [p for group in optimizer['param_groups'] for p in group['params']]
    if len(ids) != len(names) or len(set(ids)) != len(ids) or set(ids) != set(optimizer['state']):
        raise ValueError('optimizer parameter/evidence count mismatch')
    parameter_steps, zero_moments = {}, []
    for name, key in zip(names, ids):
        value = checkpoint['model_state'][name]
        state = optimizer['state'][key]
        step = int(state['step'])
        if not 0 < step <= args.expected_steps or not torch.isfinite(value).all():
            raise ValueError('invalid new parameter or optimizer step')
        for moment in ['exp_avg', 'exp_avg_sq']:
            if state[moment].shape != value.shape or not torch.isfinite(state[moment]).all():
                raise ValueError('invalid optimizer moment: '+name)
        if not (state['exp_avg_sq'] >= 0).all():
            raise ValueError('negative optimizer second moment: '+name)
        if not (state['exp_avg_sq'] > 0).any():
            zero_moments.append(name)
        parameter_steps[name] = step
    if len(zero_moments)==len(names):
        raise ValueError('no recorded learning signal in any trainable parameter')
    result = dict(state='passed', evidence_type='codec_only_checkpoint_frozen_state_and_optimizer_audit',
                  epoch=1, steps=args.expected_steps, original_tensors=len(initial['model_state']),
                  **report, trainable_parameter_tensors=len(names), parameter_optimizer_steps=parameter_steps,
                  zero_second_moment_parameter_names=zero_moments,
                  initialization_sha256=sha256(args.initialization),
                  checkpoint_sha256=sha256(args.checkpoint), manifest_sha256=sha256(args.manifest),
                  limitations='optimizer steps/moments establish actual learning updates; no AP or calibration claim')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'parameter_optimizer_steps'}))


if __name__ == '__main__':
    main()
