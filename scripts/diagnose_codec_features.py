#!/usr/bin/env python3
"""Observe real raw-cost/appearance codec distortion in sensor-only inference.

No hook changes a tensor, model state, predictions, noise or random state.
Native inference, checkpoint loading and AP output remain run_liga's work.
"""
import argparse
import json
import math
from pathlib import Path
import runpy
import sys
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def compare_features(reference, received, chunk_size=262144):
    if reference.shape != received.shape or not reference.is_floating_point() or chunk_size < 1:
        raise ValueError('equal floating-point feature shapes and positive chunk required')
    # Streaming reductions avoid materializing an entire cost volume in float64.
    left, right = reference.reshape(-1), received.reshape(-1)
    totals = dict(reference_squared=0., received_squared=0., error_squared=0.,
                  dot=0., reference_sum=0., received_sum=0.,
                  reference_negative=0., received_negative=0.)
    for offset in range(0, left.numel(), chunk_size):
        a, b = left[offset:offset+chunk_size].double(), right[offset:offset+chunk_size].double()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('nonfinite codec feature')
        values = torch.stack((a.square().sum(), b.square().sum(), (a-b).square().sum(),
            (a*b).sum(), a.sum(), b.sum(), (a < 0).sum(), (b < 0).sum())).cpu().tolist()
        for key, value in zip(totals, values):
            totals[key] += value
    count = left.numel()
    if count == 0:
        raise ValueError('empty feature')
    denominator = math.sqrt(totals['reference_squared']*totals['received_squared'])
    return dict(shape=list(reference.shape), elements=count, sums=totals,
        mse=totals['error_squared']/count,
        nmse=totals['error_squared']/totals['reference_squared'] if totals['reference_squared'] else None,
        cosine=totals['dot']/denominator if denominator else None,
        reference_rms=math.sqrt(totals['reference_squared']/count),
        received_rms=math.sqrt(totals['received_squared']/count),
        reference_mean=totals['reference_sum']/count, received_mean=totals['received_sum']/count,
        reference_negative_fraction=totals['reference_negative']/count,
        received_negative_fraction=totals['received_negative']/count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-output', type=Path, required=True)
    parser.add_argument('--expected-channel', choices=['identity','awgn'], default='identity')
    args, remainder = parser.parse_known_args()
    if not remainder or remainder[0] != 'test':
        parser.error('forward only run_liga test arguments')
    if args.feature_output.exists():
        parser.error('preserve earlier feature evidence; choose unique output')
    sys.path[:0] = [str(ROOT/'src'), str(ROOT/'third_party/LIGA-Stereo'),
                   str(ROOT/'third_party/LIGA-Stereo/tools')]
    import liga.models
    upstream_builder = liga.models.build_network
    args.feature_output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    handles = []
    with args.feature_output.open('x') as stream:
        def builder(*pargs, **kwargs):
            model = upstream_builder(*pargs, **kwargs)
            link = model.backbone_3d.semantic_link
            if link is None or model.backbone_3d.semantic_link_boundary != 'raw_cost' or link.channel.kind != args.expected_channel:
                raise ValueError('this locked diagnostic requires the specified raw-cost channel')
            def observe(module, inputs, outputs):
                if module.training:
                    raise RuntimeError('feature diagnostic cannot train')
                cost, app, batch, _ = inputs
                if cost.shape[0] != 1 or len(batch['frame_id']) != 1:
                    raise ValueError('exactly one scheduled stereo frame required')
                with torch.no_grad():
                    row = dict(frame_id=str(batch['frame_id'][0]), channel=module.channel.kind,
                        cost=compare_features(cost, outputs[0]), appearance=compare_features(app, outputs[1]))
                    # This simple reference is not an optimal reconstruction or
                    # a detection ceiling: it measures pool/interpolate alone.
                    pooled = module.cost_encoder[0](cost)
                    reference = F.interpolate(pooled, size=cost.shape[2:], mode='trilinear', align_corners=False)
                    row['cost_pool_interpolate_reference'] = compare_features(cost, reference)
                    row['cost_pooled_shape'] = list(pooled.shape)
                    del reference, pooled
                    pooled = module.appearance_encoder[0](app)
                    reference = F.interpolate(pooled, size=app.shape[2:], mode='bilinear', align_corners=False)
                    row['appearance_pool_interpolate_reference'] = compare_features(app, reference)
                    row['appearance_pooled_shape'] = list(pooled.shape)
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                stream.flush()
                records.append(row['frame_id'])
            handles.append(link.register_forward_hook(observe))
            return model
        liga.models.build_network = builder
        try:
            sys.argv = [str(ROOT/'scripts/run_liga.py')]+remainder
            runpy.run_path(str(ROOT/'scripts/run_liga.py'), run_name='__main__')
        finally:
            liga.models.build_network = upstream_builder
            for handle in handles:
                handle.remove()
    fold = json.loads((ROOT/'data/internal-tuning-fold-001.json').read_text())
    if records != fold['folds']['geocomm_tune_holdout']['ids']:
        raise ValueError('feature evidence must cover all ordered372 heldout IDs')
    print(json.dumps(dict(scope=args.expected_channel+' codec feature distortion, no training', frames=len(records))))


if __name__ == '__main__':
    main()
