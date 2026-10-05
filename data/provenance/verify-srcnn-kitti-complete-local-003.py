"""Streaming checks of every transferred SRCNN002 update and terminal artifact.

Local PNG/patch or GPU/backpropagation replay is deliberately not claimed.
View order and patch coordinates are independently regenerated from the locked
PCG64 streams, including all formal epochs; native audit replays actual pixels.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/original-paper-scenarios/srcnn-KITTI-adaptation-code-002'))
import common as c

SHAPES = {'weight1': (64, 1, 9, 9), 'bias1': (64,), 'weight2': (32, 64, 5, 5),
          'bias2': (32,), 'weight3': (1, 32, 5, 5), 'bias3': (1,)}
PLAN = ROOT / 'experiments/original-paper-scenarios/srcnn-KITTI-formal-record-validation-001.md'


def mapped(native, prefix):
    path = Path(native)
    assert path.is_absolute() and '..' not in path.parts
    if path.is_relative_to('/home/sheng/paper6'):
        return ROOT / path.relative_to('/home/sheng/paper6')
    assert path.is_relative_to('/mnt/d/paper6/runs')
    assert path.relative_to('/mnt/d/paper6/runs').parts[0] in {prefix + '-cr' + str(r) for r in (10, 30, 50)}
    return ROOT / 'data/engineering' / (prefix + '-native-transfer') / str(path).lstrip('/')


def digest_format(value):
    assert isinstance(value, str) and len(value) == 64 and all(v in '0123456789abcdef' for v in value)


def descriptor(value, shape, dtype):
    assert set(value) == {'dtype', 'shape', 'sha256'}
    assert value['dtype'] == dtype and value['shape'] == list(shape)
    digest_format(value['sha256'])


def split_check(ids, scope):
    assert all(isinstance(frame, str) and len(frame) == 6 and frame.isdecimal() for frame in ids)
    assert len(ids) == len(set(ids)) == (2 if scope == 'engineering' else 3712)
    if scope == 'engineering':
        assert ids == ['000000', '000003']
    else:
        # Exact published split identity without requiring local raw KITTI.
        assert any(hashlib.sha256((separator.join(ids) + suffix).encode()).hexdigest() == c.TRAIN_SHA
                   for separator in ('\n', '\r\n') for suffix in ('', separator))


def verify(scope):
    prefix = 'srcnn-kitti-' + scope + '-seed17-002'
    epochs, updates, views_count = (6, 6, 4) if scope == 'engineering' else (20, 37120, 7424)
    target_output = ROOT / 'data/provenance' / (prefix + ('-complete-local-003.json' if scope == 'engineering' else '-local-verification.json'))
    assert not target_output.exists()
    closure_path = ROOT / 'data/provenance' / (prefix + '-closure.json')
    closure = c.read(closure_path); source = c.sources()
    assert closure['state'] == 'closed_actual_terminal_all_three_rates_and_paired_training'
    assert closure['actual_terminal'] and closure['scope'] == scope
    assert closure['updates'] == 3 * updates and closure['rates'] == 3 and closure['actual_source_views'] == views_count
    assert closure['sources'] == source and len(closure['artifacts_sha256']) == 21
    for relative, digest in closure['artifacts_sha256'].items():
        path = Path(relative); assert not path.is_absolute() and '..' not in path.parts
        assert c.sha(ROOT / path) == digest, relative
    assert {p: str(mapped(n, prefix).relative_to(ROOT)) for p, n in closure['native_to_local_artifacts'].items()} == {
        p: p for p in closure['artifacts_sha256']}
    cycle_path = ROOT / 'data/runs' / (prefix + '-cycle.json')
    launch_path = ROOT / 'data/runs' / (prefix + '-launch.json')
    cycle, launch = c.read(cycle_path), c.read(launch_path)
    assert cycle['pid'] == launch['pid'] == closure['pid']
    assert cycle['sources'] == launch['sources'] == source
    assert cycle['state'] == 'finished_six_commands_pending_terminal_closure'
    assert cycle['scope'] == launch['scope'] == scope
    assert len(cycle['completed_commands']) == 6 and all(v['returncode'] == 0 for v in cycle['completed_commands'])
    assert c.sha(ROOT / 'data/provenance/srcnn-kitti-cycle-002.py') == cycle['controller_sha256'] == launch['controller_sha256']
    assert set(cycle['rates']) == {'10', '30', '50'}
    initial = c.states(c.initial_model()); assert set(initial) == set(SHAPES)
    expected_artifacts = {str(p.relative_to(ROOT)) for p in (cycle_path, launch_path, mapped(launch['log'], prefix))}
    for row in cycle['completed_commands']:
        path = mapped(row['log'], prefix); assert c.sha(path) == row['log_sha256']
        expected_artifacts.add(str(path.relative_to(ROOT)))
    shared_digest = shared_images = shared_ids = None
    summaries = {}; total = 0
    for rate, item in cycle['rates'].items():
        manifest, audit_path = mapped(item['manifest'], prefix), mapped(item['audit'], prefix)
        run, report = c.read(manifest), c.read(audit_path)
        assert c.sha(manifest) == item['manifest_sha256'] == report['manifest_sha256']
        assert c.sha(audit_path) == item['audit_sha256']
        assert run['state'] == 'finished' and report['state'] == 'passed_all_updates_native_independent_patches_and_final_Adam'
        assert run['scope'] == report['scope'] == scope and run['rate'] == report['rate'] == int(rate)
        assert run['epochs'] == run['selected_epoch'] == epochs
        assert run['required_updates'] == run['optimizer_steps'] == report['updates'] == updates
        assert run['sources'] == report['sources'] == source
        assert run['initial_state'] == initial and run['trainable_tensors'] == 6
        assert run['data_barrier'] and run['no_GT_calibration_detector_or_validation']
        assert not run['TF32'] and not run['mixed_precision'] and run['batch_workers'] == 0 and run['CPU_threads'] == 2
        assert run['physical_free_before_bytes'] >= 12 * 2**30
        assert run['peak_reserved_bytes'] + 2 * 2**30 <= run['physical_free_before_bytes']
        ids = run['ordered_ids']; split_check(ids, scope)
        assert shared_ids is None or ids == shared_ids; shared_ids = ids
        views = [(frame, camera) for frame in ids for camera in ('image_2', 'image_3')]
        pngs = {str(c.DATA / 'training' / camera / (frame + '.png')) for frame, camera in views}
        assert set(run['actual_source_images_sha256']) == pngs
        for value in run['actual_source_images_sha256'].values(): digest_format(value)
        assert shared_images is None or run['actual_source_images_sha256'] == shared_images
        shared_images = run['actual_source_images_sha256']
        raw, weight_path = mapped(run['raw_training_path'], prefix), mapped(run['checkpoint'], prefix)
        assert c.sha(raw) == run['raw_sha256'] == report['raw_sha256']
        assert c.sha(weight_path) == run['checkpoint_sha256'] == report['checkpoint_sha256']
        weights = torch.load(weight_path, map_location='cpu', weights_only=True)
        assert weights['initial_state'] == initial and weights['sources'] == source
        assert weights['epochs'] == epochs and weights['optimizer_steps'] == updates and weights['rate'] == int(rate)
        model = weights['model']; assert set(model) == set(SHAPES)
        assert {name: c.describe(value) for name, value in model.items()} == run['final_state'] == report['final_state']
        assert run['final_state'] != initial
        optimizer = weights['optimizer']; assert len(optimizer['param_groups']) == 1 and set(optimizer['state']) == set(range(6))
        group = optimizer['param_groups'][0]
        assert group['params'] == list(range(6)) and group['lr'] == 1e-4
        assert group['betas'] == (.9, .999) and group['eps'] == 1e-8 and group['weight_decay'] == 0
        for index, (name, parameter) in enumerate(model.items()):
            assert parameter.dtype == torch.float32 and tuple(parameter.shape) == SHAPES[name] and torch.isfinite(parameter).all()
            moment = optimizer['state'][index]
            assert set(moment) == {'step', 'exp_avg', 'exp_avg_sq'} and int(moment['step']) == updates
            for field in ('exp_avg', 'exp_avg_sq'):
                assert moment[field].shape == parameter.shape and moment[field].dtype == torch.float32
                assert torch.isfinite(moment[field]).all()
            assert (moment['exp_avg_sq'] >= 0).all()
        expected_artifacts.update(str(p.relative_to(ROOT)) for p in (manifest, audit_path, raw, weight_path))
        assert len(report['target_hashes_by_step']) == updates
        order_rng = np.random.Generator(np.random.PCG64(17)); patch_rng = np.random.Generator(np.random.PCG64(1719))
        paired = hashlib.sha256(); step = 0; epoch_stats = []; image_descriptors = {}
        with raw.open() as stream:
            for epoch in range(1, epochs + 1):
                permutation = order_rng.permutation(len(views)); losses = []
                for offset in range(0, len(views), 4):
                    line = stream.readline(); assert line.endswith('\n'); row = json.loads(line); step += 1
                    assert row['step'] == step and row['epoch'] == epoch and row['rate'] == int(rate)
                    expected_views = [views[int(index)] for index in permutation[offset:offset + 4]]
                    assert [(v['frame_id'], v['camera']) for v in row['views']] == expected_views
                    assert len(row['views']) == len(row['sampling']) == 4
                    assert row['parameter_tensors'] == 6 and set(row['gradients']) == set(row['adam_steps']) == set(initial)
                    assert set(row['adam_steps'].values()) == {step} and 0 <= row['peak_reserved_bytes'] <= run['peak_reserved_bytes']
                    assert math.isfinite(row['loss']) and row['loss'] >= 0; losses.append(row['loss'])
                    for name, value in row['gradients'].items():
                        assert math.isfinite(value['norm']) and value['norm'] >= 0
                        assert isinstance(value['nonzero'], int) and 0 <= value['nonzero'] <= math.prod(SHAPES[name])
                    descriptor(row['input_tensor'], (32, 1, 49, 49), '<f4')
                    descriptor(row['target_tensor'], (32, 1, 49, 49), '<f4')
                    assert row['target_tensor']['sha256'] == report['target_hashes_by_step'][step - 1]
                    for view, sample in zip(row['views'], row['sampling']):
                        path = str(c.DATA / 'training' / view['camera'] / (view['frame_id'] + '.png'))
                        assert view['source_path'] == path and view['source_sha256'] == shared_images[path]
                        h, w = sample['native_hw']; assert isinstance(h, int) and isinstance(w, int) and h >= 49 and w >= 49
                        descriptor(sample['native_RGB8'], (h, w, 3), '|u1')
                        assert path not in image_descriptors or image_descriptors[path] == sample['native_RGB8']
                        image_descriptors[path] = sample['native_RGB8']
                        coords = [[int(patch_rng.integers(0, h - 48)), int(patch_rng.integers(0, w - 48))] for _ in range(8)]
                        assert sample['patch_coordinates'] == coords
                    payload = dict(step=step, epoch=epoch, views=row['views'], sampling=row['sampling'], target=row['target_tensor'])
                    paired.update(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n')
                epoch_stats.append(dict(epoch=epoch, updates=len(losses), mean_loss=math.fsum(losses) / len(losses),
                                        minimum_loss=min(losses), maximum_loss=max(losses)))
            assert stream.read() == ''
        assert step == updates and len(image_descriptors) == views_count
        assert report['patches'] == updates * 32 and report['source_view_visits'] == updates * 4 and report['unique_source_views'] == views_count
        current = paired.hexdigest(); assert shared_digest is None or current == shared_digest; shared_digest = current
        total += step; summaries[rate] = dict(updates=step, final_checkpoint_sha256=run['checkpoint_sha256'],
                                            peak_reserved_bytes=run['peak_reserved_bytes'], epoch_losses_descriptive=epoch_stats)
    assert total == 3 * updates and set(closure['artifacts_sha256']) == expected_artifacts and c.sources() == source
    result = dict(state='passed_all' + str(total) + '_transferred_SRCNN_training_records_and_terminal_artifacts',
                  checked_unix=time.time(), sources=source, scope=scope, updates=total, rates=3,
                  source_views=views_count, patches=total * 32, artifacts_verified=21,
                  closure_sha256=c.sha(closure_path), acceptance_plan_sha256=c.sha(PLAN),
                  verifier_sha256=c.sha(Path(__file__)), paired_order_sampling_target_sha256=shared_digest,
                  independently_regenerated_view_and_patch_streams=True, summaries=summaries,
                  limitation='All transferred rows/weights/Adam/epoch sampling checked; full PNG/patch replay native only, no local GPU/backpropagation replay or AP')
    with target_output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(state=result['state'], scope=scope, updates=total, patches=total * 32)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--scope', choices=('engineering', 'formal'), required=True)
    verify(parser.parse_args().scope)
