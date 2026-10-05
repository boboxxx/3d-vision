"""Check every transferred F9 engineering artifact and all 24 raw native rows."""
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'stereo-epipolar-native-sanity-001'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def main():
    output = ROOT / 'data/provenance' / (PREFIX + '-local-verification-001.json')
    assert not output.exists()
    cp = ROOT / 'data/provenance' / (PREFIX + '-closure.json')
    closure = read(cp)
    assert closure['state'] == 'closed_all_four_engineering_audits_actual_terminal'
    assert closure['actual_terminal'] and closure['ps_returncode'] == 1
    assert not closure['formal_AP_started'] and closure['actual_training_records'] == 24
    for path, expected in closure['artifacts_sha256'].items(): assert sha(ROOT / path) == expected, path
    runs = ROOT / 'data/runs'; fingerprints = {}; noise = {}; profile = {}; empties = {}
    sequence = ['student', 'student', 'link_start', 'channel', 'link_done', 'build_cost', 'backbone_done', 'map_to_bev', 'BEV', 'GT_at_3D_head', 'head3D']
    for arm in ('U', 'G', 'P', 'S'):
        m = read(runs / (PREFIX + '-' + arm + '.json'))
        a = read(runs / (PREFIX + '-' + arm + '-training-audit.json'))
        raw = runs / (PREFIX + '-' + arm + '-training-audit.records/training.jsonl')
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        assert len(rows) == 6 and m['state'] == 'finished' and a['state'] == 'passed'
        assert m['full_state_tensors'] == 539 and a['trained_parameter_tensors'] == (51 if arm == 'U' else 55)
        assert a['inactive_states_identical'] == (488 if arm == 'U' else 484) and a['all_optimizer_update_counts'] == 6
        assert sha(raw) == a['raw_snapshot_sha256'] == m['training_records_sha256']
        assert a['manifest_sha256'] == sha(runs / (PREFIX + '-' + arm + '.json'))
        assert m['initialization_sha256'] == '6e54ebc5fa696db26b3902cdd129ff42544635db4cfb56dfdb7f4b94841c5193'
        assert m['peak_reserved_GiB']*2**30 + 2*2**30 <= m['gpu_free_before_bytes']
        fingerprints[arm] = []; noise[arm] = []; empties[arm] = 0
        for step, row in enumerate(rows, 1):
            assert row['step'] == step and row['optimization_scope'] == arm and row['epoch'] == 1
            assert row['sequence'] == sequence and row['GT_introduced_only_at_head']
            assert row['cost_inputs_are_received_features'] and row['appearance_input_is_received_feature'] and row['all_modules_eval']
            assert 'gt_boxes' not in row['sender_input_keys']
            expected_loss = row['classification_loss'] + row['location_loss'] + row['direction_loss'] + row['iou_weight']*row['iou_loss_raw']
            assert math.isclose(row['loss'], expected_loss, rel_tol=3e-6, abs_tol=1e-6)
            assert row['channel'] == 'awgn' and row['total_complex_uses'] == row['data_complex_uses'] == 62400
            assert row['header_complex_uses'] == row['pilot_complex_uses'] == 0
            assert row['stereo_complex_uses'] == 49920 and row['appearance_complex_uses'] == 12480
            fp = row['augmented_evidence']; value = {k: v for k, v in fp.items() if k != 'sha256'}
            assert sha_bytes(canonical(value)) == fp['sha256'] and fp['version'] == 'F9-paired-conditions-v1'
            assert row['frame_id'] == fp['frame_id'] and row['GT_shape'] == fp['arrays']['gt_boxes']['shape']
            fingerprints[arm].append(fp); noise[arm].append(row['channel_condition'])
            condition = row['channel_condition']
            assert condition['noise_seed'] == 1928 and condition['isolated_noise_generator']
            assert condition['noise_rng_before_sha256'] != condition['noise_rng_after_sha256']
            if step > 1: assert noise[arm][-2]['noise_rng_after_sha256'] == condition['noise_rng_before_sha256']
            coding = row['F9_coding']; energy = coding['stereo_group_energy']; gains = coding['amplitude_gains']
            assert coding['arm'] == arm and not coding['receiver_side_information'] and len(energy) == len(gains) == 64
            assert all(math.isfinite(v) and v >= 0 for v in energy)
            assert all(.5 <= v <= 2 for v in gains)
            assert math.isclose(sum(energy) + coding['appearance_energy'], coding['actual_energy_float64'], rel_tol=1e-12, abs_tol=1e-8)
            assert abs(coding['actual_energy_float64']/62400 - 1) <= 1e-5
            if arm == 'U' or step == 1: assert gains == [1.] * 64
            if arm in ('P', 'S'):
                assert len(coding['matrix_records']) == 2
                assert all(v['cross_vertical_mass'] == 0 and v['row_sum_max_error'] <= 6e-5 for v in coding['matrix_records'])
            else: assert coding['matrix_records'] is None
            assert set(row['parameter_gradients']) == set(m['trainable_parameter_names'])
            assert all(v['finite'] and math.isfinite(v['norm']) and v['norm'] >= 0 for v in row['parameter_gradients'].values())
            if step == 1 and arm != 'U':
                assert all(v['nonzero'] == 0 for n, v in row['parameter_gradients'].items() if 'gain_head.0.' in n)
            if row['empty_GT']:
                empties[arm] += 1
                assert row['GT_shape'] == [1, 0, 8] and row['positive_anchors'] == 0 and row['background_anchors'] > 0
                assert all(row[k] == 0 for k in ('location_loss', 'direction_loss', 'iou_loss_raw'))
        assert empties[arm] == a['summary']['empty_GT_frames'] and empties[arm] > 0
        assert all(any(row['parameter_gradients'][name]['nonzero'] > 0 for row in rows) for name in m['trainable_parameter_names'])
        profile[arm] = m['sender_profile']
    assert all(fingerprints[arm] == fingerprints['U'] and noise[arm] == noise['U'] for arm in ('G', 'P', 'S'))
    result = dict(state='passed_all_transferred_engineering_artifacts_and24_raw_rows', checked_unix=time.time(),
                  closure_sha256=sha(cp), verifier_sha256=sha(Path(__file__)), artifacts_verified=len(closure['artifacts_sha256']),
                  records=24, paired_steps=6, arms=['U', 'G', 'P', 'S'], empty_GT=empties, sender_profiles=profile,
                  formal_AP_started=False,
                  scope='Complete transferred metadata/raw-row verification; full539-state/Adam/CUDA-noise and large profile traces were checked on sheng, not rerun locally.')
    with output.open('x') as stream: json.dump(result, stream, indent=2)
    print(json.dumps(dict(state=result['state'], artifacts=result['artifacts_verified'], records=24)))


def sha_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


if __name__ == '__main__':
    main()
