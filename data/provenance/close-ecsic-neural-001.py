"""Fresh terminal/source/artifact closure, with small evidence snapshots only."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/original-paper-scenarios/ecsic-neural-reception-code'))
import contract as c

PREFIX = 'ecsic-neural-reception-native-001'


def main():
    provenance = ROOT / 'data/provenance'
    output = provenance / (PREFIX + '-closure.json')
    listing = provenance / (PREFIX + '-transfer-files-001.txt')
    snapshot = ROOT / 'data/engineering' / PREFIX
    assert not output.exists() and not listing.exists() and not snapshot.exists()
    cycle_path = ROOT / 'data/runs' / (PREFIX + '-cycle.json')
    launch_path = ROOT / 'data/runs' / (PREFIX + '-launch.json')
    audit_path = provenance / (PREFIX + '-audit.json')
    cycle, launch, audit = c.read(cycle_path), c.read(launch_path), c.read(audit_path)
    ps = subprocess.run(['ps', '-p', str(launch['pid']), '-o', 'pid,stat,etime,args'], capture_output=True, text=True)
    assert ps.returncode == 1 and len(ps.stdout.splitlines()) <= 1 and cycle['pid'] == launch['pid']
    assert cycle['state'] == 'finished_all20_outcomes11_neural_crop_pairs_and_audit_pending_terminal_closure'
    assert audit['state'] == 'passed_all20_outcomes11_causal_neural_and_exact_crops'
    assert (audit['packets'], audit['received'], audit['erased'], audit['full_arrays_exact'], audit['cropped_views_exact'], audit['distinct_successful_child_processes']) == (20, 11, 9, 198, 22, 22)
    assert audit['all225_states_exact_and_readonly'] and audit['physical_attempts_resources_unchanged'] and not audit['KITTI_AP_measured']
    assert c.sha(audit_path) == cycle['audit_sha256']
    directory = Path(cycle['output_dir'])
    manifest_path = directory / 'manifest.json'
    assert c.sha(manifest_path) == cycle['manifest_sha256'] == audit['producer_manifest_sha256']
    manifest = c.read(manifest_path)
    assert c.identities() == cycle['sources'] == manifest['sources']
    assert c.predecessor()[1] == cycle['predecessor'] == manifest['predecessor']
    assert cycle['started_unix'] <= manifest['started_unix'] <= manifest['ended_unix'] <= cycle['ended_unix'] <= time.time()
    assert len(cycle['completed_commands']) == 2
    assert [x['label'] for x in cycle['completed_commands']] == ['neural-crop', 'independent-audit']
    paths = {cycle_path, launch_path, audit_path, ROOT / 'logs' / (PREFIX + '-controller.log')}
    for command in cycle['completed_commands']:
        log = ROOT / command['log_path']
        assert command['returncode'] == 0 and c.sha(log) == command['log_sha256']
        paths.add(log)
    # Raw arrays stay on D:, but every native file is freshly read and hashed.
    assert len(audit['native_artifacts_sha256']) == 67
    for path, digest in audit['native_artifacts_sha256'].items():
        assert c.sha(path) == digest
    snapshot.mkdir()
    snapshot_map = {}
    for path, digest in audit['metadata_artifacts_sha256'].items():
        source = Path(path)
        target = snapshot / source.relative_to(directory)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        assert c.sha(target) == digest
        relative = str(target.relative_to(ROOT))
        snapshot_map[relative] = dict(native_path=path, sha256=digest)
        paths.add(target)
    assert len(snapshot_map) == 45
    for name in ('ecsic-neural-sheng-CPU-001.json',): paths.add(ROOT / 'data/engineering' / name)
    for name in ('ecsic-received-sheng-transfer-001.json',): paths.add(provenance / name)
    for p, digest in cycle['predecessor']['received_files_sha256'].items():
        assert c.sha(ROOT / p) == digest
        paths.add(ROOT / p)
    result = dict(state='closed_actual_terminal_all20_outcomes11_neural_crop_pairs',
        prefix=PREFIX, checked_unix=time.time(), actual_terminal=True,
        actual_ps=ps.stdout, ps_returncode=ps.returncode, pid=launch['pid'],
        cycle_sha256=c.sha(cycle_path), manifest_sha256=c.sha(manifest_path), audit_sha256=c.sha(audit_path),
        sources=cycle['sources'], predecessor=cycle['predecessor'], snapshot_map=snapshot_map,
        artifacts_sha256={str(p.relative_to(ROOT)): c.sha(p) for p in sorted(paths)},
        native_artifacts_sha256=audit['native_artifacts_sha256'],
        packets=20, received=11, erased=9, full_arrays_exact=198, cropped_views_exact=22,
        native_raw_arrays_replayed=True, local_raw_arrays_transferred=False, KITTI_AP_measured=False,
        scope='Fixed two-source engineering; server raw-array equality, local metadata/accepted-byte closure only')
    with output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    paths.add(output)
    with listing.open('x') as stream: stream.write('\n'.join(str(p.relative_to(ROOT)) for p in sorted(paths)) + '\n')
    print(json.dumps(dict(state=result['state'], files=len(paths), snapshots=45, raw_native_files=67)), flush=True)


if __name__ == '__main__': main()
