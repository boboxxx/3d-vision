"""Independent saved-artifact audit; imports no producer/model/channel/parser code."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
IDS = ['000036', '000054', '000071', '000082', '000113', '000141']
PARENT_SHA = '6e54ebc5fa696db26b3902cdd129ff42544635db4cfb56dfdb7f4b94841c5193'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''): h.update(part)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(str(value.dtype).encode() + str(tuple(value.shape)).encode() + value.numpy().tobytes()).hexdigest()


def archive(path, paired=False):
    keys = ('pred_boxes', 'pred_scores', 'pred_labels')
    expected = {p + k for p in ('explicit_', 'reference_') for k in keys} if paired else set(keys)
    with np.load(path, allow_pickle=False) as f:
        assert set(f.files) == expected
        values = {k: f[k].copy() for k in f.files}
    for prefix in (('explicit_', 'reference_') if paired else ('',)):
        boxes, scores, labels = (values[prefix + k] for k in keys)
        assert boxes.ndim == 2 and boxes.shape[1] == 7 and scores.shape == labels.shape == (len(boxes),)
        assert boxes.dtype == scores.dtype == np.float32 and labels.dtype.kind in 'iu'
        assert all(np.isfinite(v).all() for v in (boxes, scores, labels))
        assert (scores >= 0).all() and (scores <= 1).all() and (labels >= 1).all() and (labels <= 3).all()
    if paired:
        assert all(values['explicit_' + k].dtype == values['reference_' + k].dtype and
                   np.array_equal(values['explicit_' + k], values['reference_' + k]) for k in keys)
    return values


def replay_rng(before, after, channel):
    for item in (before, after):
        assert sha_bytes(bytes.fromhex(item['state_hex'])) == item['sha256']
        assert item['backend'] == 'cuda' and item['device'] == 'cuda:0'
    assert (before == after) == (channel == 'identity')
    generator = torch.Generator(device='cuda:0')
    generator.set_state(torch.tensor(list(bytes.fromhex(before['state_hex'])), dtype=torch.uint8))
    if channel == 'awgn': torch.randn((1, 62400, 2), device='cuda:0', dtype=torch.float32, generator=generator)
    assert generator.get_state().cpu().numpy().tobytes().hex() == after['state_hex']


def sha_bytes(value): return hashlib.sha256(value).hexdigest()


def row_audit(row, arm, channel):
    assert row['frame_id'] in IDS and row['arm'] == arm and row['channel'] == channel
    assert row['sequence'] == ['student', 'student', 'link_start', 'channel', 'link_done', 'build_cost']
    assert not row['autograd_enabled'] and set(row['sensor_input_keys']) == {'batch_size', 'left_img', 'right_img', 'frame_id', 'image_shape', 'calib'}
    assert row['cost_inputs_are_received_features'] and row['appearance_input_is_received_feature']
    assert row['channel_input_shape'] == [1, 62400, 2]
    a, c = row['accounting'], row['F9_coding']
    assert a['channel'] == channel and a['snr_db'] == c['nominal_snr'] == 10.
    assert a['allocation'] == ('uniform' if arm == 'U' else 'F9_' + arm) and c['arm'] == arm
    assert a['data_complex_uses'] == a['total_complex_uses'] == c['data_complex_uses'] == 62400
    assert a['stereo_complex_uses'] == 49920 and a['appearance_complex_uses'] == 12480
    assert a['header_complex_uses'] == a['pilot_complex_uses'] == 0
    assert a['boundary'] == 'stereo_features_before_receiver_cost' and not c['receiver_side_information']
    assert abs(a['tx_energy_per_frame'][0] / 62400 - 1) <= 1e-4
    assert c['amplitude_gains'] == [1.] * 64 and len(c['stereo_group_energy']) == 64
    assert all(math.isfinite(v) and v >= 0 for v in c['stereo_group_energy'])
    assert math.isclose(sum(c['stereo_group_energy']) + c['appearance_energy'], c['actual_energy_float64'], rel_tol=1e-12, abs_tol=1e-8)
    assert math.isclose(c['actual_energy_float64'], a['tx_energy_per_frame'][0], rel_tol=1e-5, abs_tol=1e-3)
    if arm in ('P', 'S'):
        assert len(c['matrix_records']) == 2
        assert all(m['row_sum_max_error'] <= 6e-5 and m['cross_vertical_mass'] == 0 for m in c['matrix_records'])
    else: assert c['matrix_records'] is None
    for branch in ('left_stereo', 'right_stereo', 'appearance'):
        r = row[branch]; s = r['sums']; n = math.prod(r['shape'])
        assert r['shape'] == ([1, 32, 80, 312] if branch == 'appearance' else [1, 32, 320, 1248])
        assert r['elements'] == n and all(math.isfinite(v) for v in s.values())
        assert min(s['reference_squared'], s['received_squared'], s['error_squared']) >= 0
        assert math.isclose(r['mse'], s['error_squared'] / n, rel_tol=1e-12, abs_tol=1e-12)
    replay_rng(row['noise_rng_before'], row['noise_rng_after'], channel)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cycle', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); assert not args.output.exists()
    cycle = read(args.cycle)
    assert cycle['engineering_only'] and cycle['no_AP'] and cycle['ids'] == IDS
    expected = {f'{a}-{c}10' for a in ('U', 'G', 'P', 'S') for c in ('identity', 'awgn')}
    assert set(cycle['conditions']) == expected and len(cycle['completed_commands']) == 8
    assert all(x['returncode'] == 0 and sha(ROOT / x['log_path']) == x['log_sha256'] for x in cycle['completed_commands'])
    manifest = ROOT / 'data/provenance' / ('server-source-manifest-' + cycle['prefix'] + '.json')
    assert sha(manifest) == cycle['source_manifest_sha256']
    sources = read(manifest)
    roots = dict(project=ROOT, liga=ROOT / 'third_party/LIGA-Stereo', mmdet=ROOT / 'third_party/mmdetection_kitti',
                 stereo_rcnn=ROOT / 'third_party/Stereo-RCNN', F7=ROOT, F8=ROOT, experiment=ROOT, evaluation=ROOT)
    assert set(sources) == set(roots)
    for name, entry in sources.items():
        files = entry['actual_sources']['file_hashes']
        assert all(sha(roots[name] / p) == h for p, h in files.items())
        assert sha_bytes(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()) == entry['actual_sources']['sha256']
    for p, h in {**cycle['protocols_sha256'], **cycle['CPU_gates_sha256']}.items(): assert sha(ROOT / p) == h
    parent_path = Path('/mnt/d/paper6/runs/stereo-encoder-native-seed17-001-joint/checkpoint_epoch_1.pth')
    assert sha(parent_path) == PARENT_SHA
    parent = torch.load(parent_path, map_location='cpu', weights_only=False)['model_state']; assert len(parent) == 535
    inputs = {}; rows = {}; predictions = {}; artifacts = {}; references = 0
    closure = read(ROOT / 'data/provenance/stereo-epipolar-native-sanity-001-closure.json')
    for label in sorted(expected):
        arm, channel10 = label.split('-'); channel = channel10[:-2]
        item = cycle['conditions'][label]; path = ROOT / item['report_path']
        assert sha(path) == item['report_sha256']; artifacts[str(path)] = sha(path)
        value = read(path); folder = Path(value['output_dir'])
        assert value['state'] == 'passed' and value['engineering_only'] and value['no_AP']
        assert value['arm'] == arm and value['channel'] == channel and value['snr_db'] == 10. and value['seed'] == 17
        assert value['frames'] == 6 and value['ids'] == IDS and value['parent_states_exact'] == 535
        assert value['source_manifest_sha256'] == sha(manifest)
        assert value['source_identities'] == {k: v['actual_sources'] for k, v in sources.items()}
        assert value['calls'] == dict(frames=6, student=12, codec=6, channel=6, build_cost=6, forbidden=0)
        initial = Path(f'/mnt/d/paper6/runs/stereo-epipolar-native-sanity-001-{arm}/initialization.pth')
        assert sha(initial) == value['initialization_sha256'] == closure['native_large_sha256'][arm][str(initial)]
        snapshot = torch.load(initial, map_location='cpu', weights_only=False)
        assert snapshot['version'] == 'F9-exact-F8joint-plus-four-head-states' and snapshot['seed'] == 17
        states = snapshot['model_state']; assert len(states) == 539
        assert all(k in states and v.dtype == states[k].dtype and torch.equal(v, states[k]) for k, v in parent.items())
        hashes = {k: tensor_hash(v) for k, v in states.items()}
        assert value['before_state_hashes'] == value['after_state_hashes'] == hashes
        assert value['checkpoint_loading']['state_hashes'] == hashes and value['checkpoint_loading']['required_states'] == 539
        assert value['checkpoint_loading']['role'] == 'engineering_initialization'
        extra = set(states) - set(parent)
        assert extra == {'backbone_3d.stereo_feature_link.gain_head.' + n for n in ('0.weight', '0.bias', '2.weight', '2.bias')}
        assert torch.count_nonzero(states['backbone_3d.stereo_feature_link.gain_head.2.weight']) == 0
        assert torch.count_nonzero(states['backbone_3d.stereo_feature_link.gain_head.2.bias']) == 0
        del snapshot, states
        for filename, digest in value['raw_artifacts_sha256'].items():
            assert sha(folder / filename) == digest; artifacts[str(folder / filename)] = digest
        assert set(value['raw_artifacts_sha256']) == {'dataset.log', 'records.jsonl', 'receiver-reference.npz'} | {k + '-predictions.npz' for k in IDS}
        rows[label] = [json.loads(line) for line in (folder / 'records.jsonl').read_text().splitlines()]
        assert [r['frame_id'] for r in rows[label]] == IDS
        for index, row in enumerate(rows[label]):
            row_audit(row, arm, channel)
            if index: assert row['noise_rng_before'] == rows[label][index-1]['noise_rng_after']
        reference = value['reference']
        assert reference['state'] == 'passed' and reference['all_boxes_scores_classes_exact'] and reference['original_prefix_unchanged']
        assert reference['new_sender_channel_cost_calls'] == 0 and reference['no_new_randomness']
        assert reference['RNG_before'] == reference['RNG_after']
        assert reference['auxiliary_calls'] == value['reference_calls'] == dict(dense_head_2d=1, depth_loss_head=1)
        assert sha_bytes(json.dumps(reference['prefix_value_identities'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()) == reference['prefix_identity_sha256']
        assert sha(folder / 'receiver-reference.npz') == reference['raw_arrays_sha256']
        paired = archive(folder / 'receiver-reference.npz', paired=True)
        predictions[label] = [archive(folder / (k + '-predictions.npz')) for k in IDS]
        assert all(np.array_equal(paired['explicit_' + k], predictions[label][0][k]) for k in predictions[label][0])
        inputs[label] = value['sensor_identities']; assert len(inputs[label]) == 6
        assert [r['frame_id'] for r in inputs[label]] == IDS
        references += 1
    for channel in ('identity', 'awgn'):
        baseline = 'U-' + channel + '10'
        for arm in ('G', 'P', 'S'):
            label = arm + '-' + channel + '10'
            assert inputs[label] == inputs[baseline]
            for i in range(6):
                for name in ('noise_rng_before', 'noise_rng_after', 'left_stereo', 'right_stereo', 'appearance', 'accounting'):
                    a, b = rows[label][i][name], rows[baseline][i][name]
                    if name == 'accounting': a, b = {k: v for k, v in a.items() if k != 'allocation'}, {k: v for k, v in b.items() if k != 'allocation'}
                    assert a == b, (label, i, name)
                assert all(np.array_equal(predictions[label][i][k], predictions[baseline][i][k]) for k in predictions[label][i])
    assert inputs['U-identity10'] == inputs['U-awgn10']
    result = dict(state='passed', engineering_only=True, no_AP=True, frames=48, references=references,
        actual48_RNG_advances_replayed=True, all_eight_native_auxiliary_references_exact=True,
        zero_head_all_four_arms_predictions_exact=True, all539_readonly_and535_parent_states_exact=True,
        cycle_prefix=cycle['prefix'], source_manifest_sha256=sha(manifest), artifacts_sha256=artifacts,
        checked_unix=time.time(), scope='Complete saved prediction/record/state/source audit and independent CUDA RNG advance replay; prefix tensors are producer fingerprints, not raw-feature replay')
    with args.output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(state='passed', frames=48, references=8)), flush=True)


if __name__ == '__main__': main()
