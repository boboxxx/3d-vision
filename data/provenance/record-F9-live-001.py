"""Read-only current F9 process/complete-row/source snapshot, never restarts work."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/geometry-link/F9/evaluation'))
from integration import source_specs
from common import source_identity
PREFIX = 'stereo-epipolar-native-seed17-001'


def read(path):
    blob = Path(path).read_bytes()
    return json.loads(blob), hashlib.sha256(blob).hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    assert not args.output.exists()
    launch, _ = read(ROOT / 'data/runs' / (PREFIX + '-launch.json'))
    cycle, cycle_sha = read(ROOT / 'data/runs' / (PREFIX + '-cycle.json'))
    ps = subprocess.run(['ps', '-p', str(launch['pid']), '-o', 'pid,stat,etime,args'], text=True, capture_output=True)
    stages = {}
    for arm in ('U', 'G', 'P', 'S'):
        manifest = ROOT / 'data/runs' / f'{PREFIX}-{arm}.json'
        if not manifest.exists(): continue
        value, manifest_sha = read(manifest)
        records = Path(value['output_dir']) / 'training.jsonl'
        raw = records.read_bytes() if records.exists() else b''
        lines = raw.splitlines(keepends=True)
        rows = [json.loads(line) for line in lines if line.endswith(b'\n')]
        stages[arm] = dict(state=value['state'], manifest_sha256=manifest_sha,
                          reported_optimizer_steps=value['optimizer_steps'], complete_raw_rows=len(rows),
                          last_actual_completed_step=rows[-1]['step'] if rows else None,
                          incomplete_trailing_record=bool(lines and not lines[-1].endswith(b'\n')))
    sources = {k: source_identity(*v) for k, v in source_specs().items()}
    assert sources == cycle['source_identities'], 'active eight-tree source identity changed'
    value = dict(state='live_readonly_snapshot', prefix=PREFIX, checked_unix=time.time(), pid=launch['pid'],
                 actual_ps=ps.stdout, ps_returncode=ps.returncode, cycle_state=cycle['state'], cycle_sha256=cycle_sha,
                 current_stage=cycle.get('current_stage'), current_arm=cycle.get('current_arm'),
                 complete_commands=len(cycle['completed_commands']), completed_endpoints=list(cycle['evaluations']),
                 training=stages, eight_tree_sources_match=True,
                 NVIDIA_memory=subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader'], text=True).strip(),
                 scope='Live progress only; incomplete runs are not final results, no AP interpretation or restart')
    with args.output.open('x') as stream: json.dump(value, stream, indent=2)
    print(json.dumps(value))


if __name__ == '__main__': main()
