"""Post-terminal engineering closure; never starts or repeats native training."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'experiments/geometry-link/F9/code')]
from geocomm.evidence import source_identity
from conditions import paired_records
from train import source_specs


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    assert args.prefix == 'stereo-epipolar-native-sanity-001'
    runs = ROOT / 'data/runs'; provenance = ROOT / 'data/provenance'
    output = provenance / (args.prefix + '-closure.json')
    assert not output.exists()
    launch_path = runs / (args.prefix + '-launch.json'); cycle_path = runs / (args.prefix + '-cycle.json')
    launch, cycle = read(launch_path), read(cycle_path)
    actual_ps = subprocess.run(['ps', '-p', str(launch['pid']), '-o', 'pid,stat,etime,args'], text=True, capture_output=True)
    assert actual_ps.returncode == 1 and len(actual_ps.stdout.splitlines()) <= 1, 'actual cycle process still exists'
    assert cycle['pid'] == launch['pid'] and cycle['engineering'] and not cycle['native_AP_started']
    assert cycle['state'] == 'finished_all_four_engineering_audits_pending_terminal_closure' and cycle['updates_per_arm'] == 6
    assert len(cycle['completed_commands']) == 8 and set(cycle['arms']) == {'U', 'G', 'P', 'S'}
    source_path = provenance / ('server-source-manifest-' + args.prefix + '.json')
    sources = read(source_path)
    assert sha(source_path) == cycle['source_manifest_sha256']
    assert all(source_identity(*value) == sources[key]['actual_sources'] for key, value in source_specs().items())
    artifacts = {launch_path, cycle_path, source_path, ROOT / 'logs' / (args.prefix + '-controller.log')}
    records = {}; arm_summary = {}; native_large = {}
    for arm in ('U', 'G', 'P', 'S'):
        item = cycle['arms'][arm]
        mp, ap = Path(item['manifest_path']), Path(item['audit_path'])
        assert sha(mp) == item['manifest_sha256'] and sha(ap) == item['audit_sha256']
        m, a = read(mp), read(ap)
        assert m['state'] == 'finished' and m['arm'] == a['arm'] == arm and m['optimizer_steps'] == a['steps'] == 6
        assert a['state'] == 'passed' and a['evidence_type'] == 'engineering_only'
        assert a['trained_parameter_tensors'] == (51 if arm == 'U' else 55)
        assert a['inactive_states_identical'] == (488 if arm == 'U' else 484)
        assert a['all_optimizer_update_counts'] == 6 and a['summary']['empty_GT_frames'] > 0
        assert a['manifest_sha256'] == sha(mp)
        assert a['checkpoint_sha256'] == m['checkpoint_sha256'] == sha(m['checkpoint_path'])
        assert a['full_initialization_sha256'] == m['full_initialization_sha256'] == sha(m['full_initialization_path'])
        assert m['full_state_tensors'] == 539 and m['source_manifest_sha256'] == sha(source_path)
        assert m['peak_reserved_GiB'] * 2**30 + 2*2**30 <= m['gpu_free_before_bytes']
        assert sha(m['sender_profile']['profiler_trace_path']) == m['sender_profile']['profiler_trace_sha256']
        raw = Path(m['output_dir']) / 'training.jsonl'; snapshot = ap.with_suffix('.records') / 'training.jsonl'
        assert sha(raw) == sha(snapshot) == m['training_records_sha256'] == a['raw_snapshot_sha256']
        records[arm] = snapshot
        artifacts.update((mp, ap, snapshot))
        native_large[arm] = {str(Path(m[k])): sha(m[k]) for k in ('checkpoint_path', 'full_initialization_path')}
        native_large[arm][m['sender_profile']['profiler_trace_path']] = m['sender_profile']['profiler_trace_sha256']
        arm_summary[arm] = dict(selected=a['trained_parameter_tensors'], fixed=a['inactive_states_identical'],
                               empty_GT=a['summary']['empty_GT_frames'], peak_reserved_GiB=m['peak_reserved_GiB'],
                               sender_profile=m['sender_profile'])
    pair_path = runs / (args.prefix + '-pair-audit.json')
    pair = read(pair_path)
    assert sha(pair_path) == cycle['paired_audit_sha256'] and pair == paired_records(records, 6)
    artifacts.add(pair_path)
    for item in cycle['completed_commands']:
        path = ROOT / item['log_path']; assert sha(path) == item['log_sha256']; artifacts.add(path)
    for name in ('F9-coupling-sheng-CPU-001.json', 'F9-training-sheng-CPU-001.json'):
        artifacts.add(ROOT / 'data/engineering' / name)
    result = dict(state='closed_all_four_engineering_audits_actual_terminal', prefix=args.prefix,
                  checked_unix=time.time(), actual_terminal=True, actual_ps=actual_ps.stdout, ps_returncode=actual_ps.returncode,
                  cycle_pid=launch['pid'], engineering_updates_per_arm=6, actual_training_records=24, arms=arm_summary,
                  source_manifest_sha256=sha(source_path), paired_audit_sha256=sha(pair_path),
                  artifacts_sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted(artifacts)}, native_large_sha256=native_large,
                  formal_AP_started=False, engineering_weights_excluded_from_formal_parent=True,
                  scope='Post-terminal full native saved-state/Adam/input-noise/record/source closure; no AP or project completion.')
    with output.open('x') as stream: json.dump(result, stream, indent=2)
    transfer = provenance / (args.prefix + '-transfer-files-001.txt')
    transfer.write_text('\n'.join([*result['artifacts_sha256'], str(output.relative_to(ROOT))]) + '\n')
    print(json.dumps(dict(state=result['state'], artifacts=len(artifacts), records=24)))


if __name__ == '__main__':
    main()
