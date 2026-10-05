"""Terminal closure and full small-artifact snapshot; never repeats inference."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/geometry-link/F9/evaluation'))
from integration import source_specs
from common import source_identity

PREFIX = 'stereo-epipolar-evaluation-integration-001'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''): h.update(part)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def main():
    output = ROOT / 'data/provenance' / (PREFIX + '-closure.json')
    transfer = ROOT / 'data/provenance' / (PREFIX + '-transfer-files-001.txt')
    assert not output.exists() and not transfer.exists()
    runs = ROOT / 'data/runs'
    launch_path = runs / (PREFIX + '-launch.json'); cycle_path = runs / (PREFIX + '-cycle.json')
    launch, cycle = read(launch_path), read(cycle_path)
    ps = subprocess.run(['ps', '-p', str(launch['pid']), '-o', 'pid,stat,etime,args'], text=True, capture_output=True)
    assert ps.returncode == 1 and len(ps.stdout.splitlines()) <= 1, 'controller still present'
    assert cycle['pid'] == launch['pid']
    assert cycle['state'] == 'finished_all48_native_frames_eight_references_and_audit_pending_terminal_closure'
    assert cycle['engineering_only'] and cycle['no_AP'] and cycle['frames'] == 48 and cycle['references'] == 8
    assert len(cycle['conditions']) == 8 and len(cycle['completed_commands']) == 9
    audit_path = runs / (PREFIX + '-audit.json'); audit = read(audit_path)
    assert audit['state'] == 'passed' and audit['frames'] == 48 and audit['references'] == 8
    assert sha(audit_path) == cycle['audit_sha256']
    source_path = ROOT / 'data/provenance' / ('server-source-manifest-' + PREFIX + '.json')
    assert sha(source_path) == cycle['source_manifest_sha256'] == audit['source_manifest_sha256']
    sources = read(source_path)
    assert {k: source_identity(*v) for k, v in source_specs().items()} == {k: v['actual_sources'] for k, v in sources.items()}
    artifacts = {launch_path, cycle_path, audit_path, source_path, ROOT / 'logs' / (PREFIX + '-controller.log')}
    artifacts.update(ROOT / p for p in cycle['CPU_gates_sha256'])
    for p, h in {**cycle['CPU_gates_sha256'], **cycle['protocols_sha256']}.items(): assert sha(ROOT / p) == h
    for command in cycle['completed_commands']:
        path = ROOT / command['log_path']; assert command['returncode'] == 0 and sha(path) == command['log_sha256']; artifacts.add(path)
    snapshots = {}; conditions = {}
    for label, item in cycle['conditions'].items():
        report_path = ROOT / item['report_path']; assert sha(report_path) == item['report_sha256']; artifacts.add(report_path)
        report = read(report_path); assert report['state'] == 'passed' and report['frames'] == 6
        assert report['before_state_hashes'] == report['after_state_hashes'] and len(report['before_state_hashes']) == 539
        folder = Path(report['output_dir']); destination = runs / (PREFIX + '-raw') / label
        assert not destination.exists(); destination.mkdir(parents=True)
        for filename, h in report['raw_artifacts_sha256'].items():
            native = folder / filename; assert sha(native) == h == audit['artifacts_sha256'][str(native)]
            snapshot = destination / filename; shutil.copyfile(native, snapshot)
            assert sha(snapshot) == h; artifacts.add(snapshot)
            snapshots[str(native)] = dict(snapshot=str(snapshot.relative_to(ROOT)), sha256=h)
        conditions[label] = dict(frames=6, reference_exact=report['reference']['all_boxes_scores_classes_exact'],
                                 all539_readonly=True, parent535_exact=True, checkpoint_sha256=report['initialization_sha256'],
                                 peak_reserved_GiB=report['peak_reserved_GiB'])
    result = dict(state='closed_all48_native_frames_actual_terminal', prefix=PREFIX, checked_unix=time.time(),
        actual_terminal=True, actual_ps=ps.stdout, ps_returncode=ps.returncode, cycle_pid=launch['pid'],
        engineering_only=True, no_AP=True, frames=48, references=8, conditions=conditions,
        independent_audit_sha256=sha(audit_path), source_manifest_sha256=sha(source_path),
        artifacts_sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted(artifacts)}, snapshots=snapshots,
        scope='Complete terminal/small raw prediction-record/state evidence closure; large native weights stay on sheng')
    with output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    transfer.write_text('\n'.join([*result['artifacts_sha256'], str(output.relative_to(ROOT))]) + '\n')
    print(json.dumps(dict(state=result['state'], artifacts=len(artifacts), frames=48, references=8)))


if __name__ == '__main__': main()
