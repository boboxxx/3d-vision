"""Independent local complete-transfer/prediction equality check; no native rerun."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'stereo-epipolar-evaluation-integration-001'
IDS = ['000036', '000054', '000071', '000082', '000113', '000141']


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def main():
    output = ROOT / 'data/provenance' / (PREFIX + '-local-verification-001.json')
    assert not output.exists()
    closure_path = ROOT / 'data/provenance' / (PREFIX + '-closure.json')
    c = read(closure_path)
    assert c['state'] == 'closed_all48_native_frames_actual_terminal' and c['actual_terminal'] and c['ps_returncode'] == 1
    assert c['engineering_only'] and c['no_AP'] and c['frames'] == 48 and c['references'] == 8
    for p, h in c['artifacts_sha256'].items(): assert sha(ROOT / p) == h, p
    audit_path = ROOT / 'data/runs' / (PREFIX + '-audit.json'); audit = read(audit_path)
    assert sha(audit_path) == c['independent_audit_sha256'] and audit['state'] == 'passed'
    assert audit['frames'] == 48 and audit['references'] == 8
    cycle = read(ROOT / 'data/runs' / (PREFIX + '-cycle.json'))
    assert cycle['pid'] == c['cycle_pid'] and len(cycle['conditions']) == 8 and len(cycle['completed_commands']) == 9
    results = {}; sensors = {}; rows = {}; count = 0
    for label, item in cycle['conditions'].items():
        path = ROOT / item['report_path']; assert sha(path) == item['report_sha256']
        report = read(path); raw = ROOT / 'data/runs' / (PREFIX + '-raw') / label
        assert report['state'] == 'passed' and report['frames'] == 6 and report['ids'] == IDS
        assert report['before_state_hashes'] == report['after_state_hashes'] and len(report['before_state_hashes']) == 539
        assert report['checkpoint_loading']['role'] == 'engineering_initialization' and report['parent_states_exact'] == 535
        assert report['reference']['RNG_before'] == report['reference']['RNG_after']
        assert report['reference_calls'] == dict(dense_head_2d=1, depth_loss_head=1)
        for filename, h in report['raw_artifacts_sha256'].items():
            assert sha(raw / filename) == h
            native = str(Path(report['output_dir']) / filename)
            assert c['snapshots'][native]['sha256'] == h == audit['artifacts_sha256'][native]
            assert c['snapshots'][native]['snapshot'] == str((raw / filename).relative_to(ROOT))
        rows[label] = [json.loads(s) for s in (raw / 'records.jsonl').read_text().splitlines()]
        assert [r['frame_id'] for r in rows[label]] == IDS
        for r in rows[label]:
            for name in ('noise_rng_before', 'noise_rng_after'):
                assert hashlib.sha256(bytes.fromhex(r[name]['state_hex'])).hexdigest() == r[name]['sha256']
        results[label] = []
        for frame in IDS:
            with np.load(raw / (frame + '-predictions.npz'), allow_pickle=False) as f:
                assert set(f.files) == {'pred_boxes', 'pred_scores', 'pred_labels'}
                result = {k: f[k].copy() for k in f.files}
                assert all(np.isfinite(v).all() for v in result.values())
                results[label].append(result); count += 1
        with np.load(raw / 'receiver-reference.npz', allow_pickle=False) as f:
            assert set(f.files) == {prefix + k for prefix in ('explicit_', 'reference_') for k in results[label][0]}
            assert all(np.array_equal(f['explicit_' + k], f['reference_' + k]) and
                       np.array_equal(f['explicit_' + k], results[label][0][k]) for k in results[label][0])
        sensors[label] = report['sensor_identities']
    for channel in ('identity', 'awgn'):
        base = 'U-' + channel + '10'
        for arm in ('G', 'P', 'S'):
            label = arm + '-' + channel + '10'
            assert sensors[label] == sensors[base]
            for i in range(6):
                assert rows[label][i]['noise_rng_before'] == rows[base][i]['noise_rng_before']
                assert rows[label][i]['noise_rng_after'] == rows[base][i]['noise_rng_after']
                assert all(np.array_equal(results[label][i][k], results[base][i][k]) for k in results[label][i])
    assert count == 48 and sensors['U-identity10'] == sensors['U-awgn10']
    value = dict(state='passed', artifacts=len(c['artifacts_sha256']), actual_prediction_frames=48, exact_receiver_references=8,
                 closure_sha256=sha(closure_path), independent_native_audit_sha256=sha(audit_path), checked_unix=time.time(),
                 scope='All transferred raw small records/predictions checked; native weights and CUDA RNG replay audited on server only; no AP')
    with output.open('x') as stream: json.dump(value, stream, indent=2)
    print(json.dumps(value))


if __name__ == '__main__': main()
