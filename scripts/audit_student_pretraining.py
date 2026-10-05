#!/usr/bin/env python3
"""Independent F1 saved-state, optimizer, scalar and fold coverage audit.

No teacher activations are archived: scalar audit checks consistency, not fresh
feature recomputation. Native detection AP is a separate required audit.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from geocomm.evidence import sha256

PREFIX = 'backbone_3d.student_semantic_link_encoder.'
INTERFACES = ('left_stereo', 'right_stereo', 'left_appearance')
FIELDS = ('student_rms', 'teacher_rms', 'normalized_mse', 'cosine')


def check(condition, message):
    if not condition:
        raise ValueError(message)


def frozen_original(reference, actual):
    converted = []
    for key, value in reference.items():
        check(key in actual, 'missing original tensor: '+key)
        if value.shape != actual[key].shape:
            check(key.startswith('lidar_model.backbone_3d.') and key.endswith('.weight')
                  and value.ndim == 5, 'unexpected layout change: '+key)
            value = value.permute(4, 0, 1, 2, 3).contiguous()
            converted.append(key)
        check(value.shape == actual[key].shape and value.dtype == actual[key].dtype
              and torch.equal(value, actual[key]), 'original tensor value changed: '+key)
    check(all(key in reference or key.startswith(PREFIX) for key in actual),
          'checkpoint contains an unscoped extra tensor')
    return converted


def check_interfaces(row):
    check(set(row) == set(INTERFACES), 'feature interface fields differ')
    for values in row.values():
        check(set(values) == set(FIELDS), 'feature metric fields differ')
        check(all(math.isfinite(v) for v in values.values()), 'nonfinite feature metric')
        check(-1.000001 <= values['cosine'] <= 1.000001, 'invalid cosine')
        check(all(values[k] >= 0 for k in FIELDS if k != 'cosine'), 'negative energy/error')


def scalar_records(path, train_ids, expected_steps, engineering):
    losses, norms, epochs = [], [], {}
    with path.open() as stream:
        for step, line in enumerate(stream, 1):
            row = json.loads(line)
            check(row['step'] == step and row['epoch'] == (step-1)//3340+1,
                  'noncontiguous update/epoch records')
            check(row['frame_id'] in train_ids, 'training frame outside locked fold')
            check(math.isfinite(row['loss']) and row['loss'] >= 0, 'nonfinite/negative training loss')
            check(math.isfinite(row['preclip_grad_norm']) and row['preclip_grad_norm'] >= 0,
                  'nonfinite/negative recorded gradient')
            check_interfaces(row['interfaces'])
            metric_loss = sum(row['interfaces'][name]['normalized_mse'] for name in INTERFACES)/3
            check(math.isclose(row['loss'], metric_loss, rel_tol=2e-6, abs_tol=1e-7),
                  'feature metrics disagree with optimized normalized MSE')
            epochs.setdefault(row['epoch'], []).append(row['frame_id'])
            losses.append(row['loss'])
            norms.append(row['preclip_grad_norm'])
    check(len(losses) == expected_steps and len(losses) > 0, 'scalar update count differs')
    check(engineering or all(len(ids) == 3340 for ids in epochs.values()), 'partial training epoch')
    return dict(steps=len(losses), first100_loss_median=statistics.median(losses[:100]),
        last100_loss_median=statistics.median(losses[-100:]), max_preclip_gradient=max(norms),
        epoch_unique_returned_frames={str(k): len(set(v)) for k, v in epochs.items()})


def holdout_records(path, ids, recorded, engineering):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    actual_ids = [row['frame_id'] for row in rows]
    check(actual_ids == (ids[:len(rows)] if engineering else ids), 'holdout coverage/order differs')
    check(len(rows) == recorded['frames'] and len(rows) > 0, 'holdout count differs')
    check(sha256(path) == recorded['records_sha256'], 'holdout records identity differs')
    for row in rows:
        check_interfaces(row['interfaces'])
        check(math.isfinite(row['loss']) and math.isclose(row['loss'],
            sum(row['interfaces'][name]['normalized_mse'] for name in INTERFACES)/3,
            rel_tol=2e-6, abs_tol=1e-7), 'holdout feature loss differs')
    for name in INTERFACES:
        for key in FIELDS:
            average = sum(row['interfaces'][name][key] for row in rows)/len(rows)
            check(math.isclose(average, recorded['interfaces'][name][key], rel_tol=1e-9, abs_tol=1e-12),
                  'holdout metric aggregation differs')
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--initialization', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--fold', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--engineering-sanity', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('preserve previous audits; unique output required')
    run = json.loads(args.manifest.read_text())
    fold = json.loads(args.fold.read_text())
    sources = json.loads(args.source_manifest.read_text())
    check(run['state'] == 'finished', 'training not finished')
    check(run['evidence_type'] == ('engineering_only' if args.engineering_sanity else
                                  'exploratory_feature_pretraining'), 'evidence type differs')
    for path, key in ((args.initialization, 'initialization_sha256'), (args.config, 'config_sha256'),
                      (args.protocol, 'protocol_sha256'), (args.fold, 'fold_sha256')):
        check(sha256(path) == run[key], 'input identity differs: '+key)
    for name, key in (('project', 'project_sources'), ('liga', 'detector_sources'), ('mmdet', 'mmdet_sources')):
        check(sources[name]['actual_sources'] == run[key], 'frozen source identity differs: '+name)
    check(run['seed'] == 17 and run['budget'] == dict(epochs=5, train_frames=3340, updates=16700,
        holdout_frames=372, batch_size=1, workers=4, lr=.001, weight_decay=.0001, clip_norm=10), 'F1 budget differs')
    train_ids = set(fold['folds']['geocomm_tune_train']['ids'])
    holdout_ids = fold['folds']['geocomm_tune_holdout']['ids']
    check(len(train_ids) == 3340 and len(holdout_ids) == 372 and not train_ids.intersection(holdout_ids), 'fold differs')
    reference = torch.load(args.initialization, map_location='cpu', weights_only=False)['model_state']
    check(len(reference) == 484, 'original tensor count differs')
    names = run['trainable_parameter_names']
    check(names and len(names) == len(set(names)) and all(name.startswith(PREFIX) for name in names),
          'trainable parameters outside student')
    summaries = []
    output = Path(run['output_dir'])
    for index, epoch in enumerate(run['epochs'], 1):
        check(epoch['epoch'] == index, 'epoch sequence differs')
        checkpoint_path = Path(epoch['checkpoint_path'])
        check(sha256(checkpoint_path) == epoch['checkpoint_sha256'], 'checkpoint identity differs')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        steps = sum(e['updates'] for e in run['epochs'][:index])
        check(checkpoint['it'] == steps and checkpoint['epoch'] == (0 if args.engineering_sanity else index),
              'checkpoint epoch/update count differs')
        converted = frozen_original(reference, checkpoint['model_state'])
        optimizer = checkpoint['optimizer_state']
        groups = optimizer['param_groups']
        check(len(groups) == 1 and groups[0]['lr'] == .001 and groups[0]['weight_decay'] == .0001,
              'optimizer configuration differs')
        param_ids = groups[0]['params']
        check(len(param_ids) == len(names) == len(set(param_ids))
              and set(param_ids) == set(optimizer['state']), 'optimizer parameter count differs')
        for name, key in zip(names, param_ids):
            parameter = checkpoint['model_state'][name]
            state = optimizer['state'][key]
            check(torch.isfinite(parameter).all() and int(state['step']) == steps, 'student update counter differs')
            for moment in ('exp_avg', 'exp_avg_sq'):
                check(state[moment].shape == parameter.shape and torch.isfinite(state[moment]).all(),
                      'invalid optimizer moment: '+name)
            check((state['exp_avg_sq'] >= 0).all() and (state['exp_avg_sq'] > 0).any(),
                  'no recorded learning signal: '+name)
        frames = holdout_records(output/f'feature_holdout_epoch_{index}.jsonl', holdout_ids,
                                 epoch['holdout'], args.engineering_sanity)
        summaries.append(dict(epoch=index, optimizer_steps=steps, original_tensors_identical=484,
            value_preserving_sparse_layout_keys=converted, holdout_frames=frames,
            checkpoint_sha256=epoch['checkpoint_sha256'], interfaces=epoch['holdout']['interfaces']))
    expected_steps = run['optimizer_steps'] if args.engineering_sanity else 16700
    check(args.engineering_sanity or (run['completed_epochs'] == 5 and len(summaries) == 5), 'incomplete five-epoch budget')
    scalars = output/'training.jsonl'
    check(sha256(scalars) == run['training_records_sha256'], 'training record identity differs')
    summary = scalar_records(scalars, train_ids, expected_steps, args.engineering_sanity)
    result = dict(state='passed', evidence_type=run['evidence_type'],
        manifest_sha256=sha256(args.manifest), source_manifest_sha256=sha256(args.source_manifest),
        initialization_sha256=sha256(args.initialization), scalar_summary=summary, epochs=summaries,
        trainable_parameter_tensors=len(names), limitations='checks saved state/updates/metrics; no fresh feature recomputation or AP claim')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
