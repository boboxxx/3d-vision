"""All transferred SRCNN engineering rows/checkpoints/Adam/terminal files."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'srcnn-kitti-engineering-seed17-002'
sys.path.insert(0, str(ROOT / 'experiments/original-paper-scenarios/srcnn-KITTI-adaptation-code-002'))
import common as c


def mapped(native):
    path = Path(native)
    if path.is_relative_to('/home/sheng/paper6'):
        return ROOT / path.relative_to('/home/sheng/paper6')
    assert path.is_relative_to('/mnt/d/paper6/runs')
    assert path.relative_to('/mnt/d/paper6/runs').parts[0].startswith(PREFIX + '-cr')
    return ROOT / 'data/engineering' / (PREFIX + '-native-transfer') / str(path).lstrip('/')


def main():
    output = ROOT / 'data/provenance' / (PREFIX + '-local-verification.json')
    assert not output.exists()
    closure_path = ROOT / 'data/provenance' / (PREFIX + '-closure.json')
    closure = c.read(closure_path)
    assert closure['state'] == 'closed_actual_terminal_all_three_rates_and_paired_training'
    assert closure['actual_terminal'] and closure['scope'] == 'engineering'
    assert closure['updates'] == 18 and closure['rates'] == 3 and closure['actual_source_views'] == 4
    assert len(closure['artifacts_sha256']) == 21 and closure['sources'] == c.sources()
    for relative, digest in closure['artifacts_sha256'].items():
        assert c.sha(ROOT / relative) == digest, relative
    assert {p: str(mapped(n).relative_to(ROOT)) for p, n in closure['native_to_local_artifacts'].items()} == {
        p: p for p in closure['artifacts_sha256']}
    cycle_path = ROOT / 'data/runs' / (PREFIX + '-cycle.json')
    launch_path = ROOT / 'data/runs' / (PREFIX + '-launch.json')
    cycle, launch = c.read(cycle_path), c.read(launch_path)
    assert cycle['pid'] == launch['pid'] == closure['pid']
    assert cycle['sources'] == launch['sources'] == closure['sources']
    assert cycle['state'] == 'finished_six_commands_pending_terminal_closure'
    assert len(cycle['completed_commands']) == 6 and all(row['returncode'] == 0 for row in cycle['completed_commands'])
    assert c.sha(ROOT / 'data/provenance/srcnn-kitti-cycle-002.py') == cycle['controller_sha256'] == launch['controller_sha256']
    assert set(cycle['rates']) == {'10', '30', '50'}
    initial = c.states(c.initial_model())
    expected_artifacts = {str(cycle_path.relative_to(ROOT)), str(launch_path.relative_to(ROOT)),
                          str(mapped(launch['log']).relative_to(ROOT))}
    for command in cycle['completed_commands']:
        path = mapped(command['log']); assert c.sha(path) == command['log_sha256']
        expected_artifacts.add(str(path.relative_to(ROOT)))
    shared = None; images = None; total = 0
    summaries = {}
    for rate, item in cycle['rates'].items():
        manifest, audit_path = mapped(item['manifest']), mapped(item['audit'])
        run, report = c.read(manifest), c.read(audit_path)
        assert c.sha(manifest) == item['manifest_sha256'] == report['manifest_sha256']
        assert c.sha(audit_path) == item['audit_sha256']
        assert run['state'] == 'finished' and report['state'] == 'passed_all_updates_native_independent_patches_and_final_Adam'
        assert run['scope'] == report['scope'] == 'engineering' and run['rate'] == report['rate'] == int(rate)
        assert run['epochs'] == run['selected_epoch'] == run['optimizer_steps'] == report['updates'] == 6
        assert run['ordered_ids'] == ['000000', '000003']
        assert run['sources'] == report['sources'] == closure['sources']
        assert run['initial_state'] == initial and run['trainable_tensors'] == 6
        assert run['data_barrier'] and run['no_GT_calibration_detector_or_validation']
        assert not run['TF32'] and not run['mixed_precision']
        assert run['peak_reserved_bytes'] + 2 * 2**30 <= run['physical_free_before_bytes']
        raw, checkpoint_path = mapped(run['raw_training_path']), mapped(run['checkpoint'])
        assert c.sha(raw) == run['raw_sha256'] == report['raw_sha256']
        assert c.sha(checkpoint_path) == run['checkpoint_sha256'] == report['checkpoint_sha256']
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        assert checkpoint['initial_state'] == initial and checkpoint['sources'] == closure['sources']
        assert checkpoint['epochs'] == checkpoint['optimizer_steps'] == 6 and checkpoint['rate'] == int(rate)
        assert {name: c.describe(value) for name, value in checkpoint['model'].items()} == run['final_state'] == report['final_state']
        assert set(checkpoint['model']) == set(initial) and run['final_state'] != initial
        optimizer = checkpoint['optimizer']
        assert len(optimizer['param_groups']) == 1 and set(optimizer['state']) == set(range(6))
        group = optimizer['param_groups'][0]
        assert group['params'] == list(range(6)) and group['lr'] == 1e-4
        assert group['betas'] == (.9, .999) and group['eps'] == 1e-8 and group['weight_decay'] == 0
        for index, parameter in enumerate(checkpoint['model'].values()):
            moment = optimizer['state'][index]
            assert set(moment) == {'step', 'exp_avg', 'exp_avg_sq'} and int(moment['step']) == 6
            assert moment['exp_avg'].shape == moment['exp_avg_sq'].shape == parameter.shape
            assert torch.isfinite(parameter).all() and torch.isfinite(moment['exp_avg']).all() and torch.isfinite(moment['exp_avg_sq']).all()
        expected_artifacts.update(str(path.relative_to(ROOT)) for path in (manifest, audit_path, raw, checkpoint_path))
        text = raw.read_text(); assert text.endswith('\n')
        rows = [json.loads(line) for line in text.splitlines()]; assert len(rows) == 6
        expected_order = list(c.order(['000000', '000003'], 6))
        paired = []
        for row, (epoch, step, views) in zip(rows, expected_order):
            assert row['epoch'] == epoch and row['step'] == step and row['rate'] == int(rate)
            assert [(item['frame_id'], item['camera']) for item in row['views']] == views
            assert row['parameter_tensors'] == 6 and set(row['gradients']) == set(row['adam_steps']) == set(initial)
            assert set(row['adam_steps'].values()) == {step}
            assert math.isfinite(row['loss']) and row['loss'] >= 0
            for descriptor in (row['input_tensor'], row['target_tensor']):
                assert descriptor['dtype'] == '<f4' and descriptor['shape'] == [32, 1, 49, 49]
            for name, value in row['gradients'].items():
                count = math.prod(initial[name]['shape'])
                assert math.isfinite(value['norm']) and value['norm'] >= 0 and 0 <= value['nonzero'] <= count
            paired.append(dict(epoch=epoch, step=step, views=row['views'], sampling=row['sampling'], target=row['target_tensor']))
            total += 1
        assert report['target_hashes_by_step'] == [row['target_tensor']['sha256'] for row in rows]
        assert report['patches'] == 192 and report['source_view_visits'] == 24 and report['unique_source_views'] == 4
        assert shared is None or shared == paired; shared = paired
        assert images is None or images == run['actual_source_images_sha256']; images = run['actual_source_images_sha256']
        summaries[rate] = dict(initial_loss=rows[0]['loss'], final_loss=rows[-1]['loss'],
                               peak_reserved_bytes=run['peak_reserved_bytes'], final_checkpoint_sha256=run['checkpoint_sha256'])
    assert total == 18 and len(images) == 4 and set(closure['artifacts_sha256']) == expected_artifacts
    result = dict(state='passed_all18_transferred_SRCNN_training_records_and_terminal_artifacts', checked_unix=time.time(),
                  sources=closure['sources'], updates=total, rates=3, source_views=4, patches=576,
                  artifacts_verified=21, closure_sha256=c.sha(closure_path), summaries=summaries,
                  limitation='All actual transferred rows/weights/Adam checked; full PNG/patch replay native only, no local GPU replay or AP')
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(state=result['state'], updates=total, patches=576)))


if __name__ == '__main__':
    main()
