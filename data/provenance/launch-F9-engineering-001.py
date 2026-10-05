"""One unique detached sheng engineering cycle; no automatic restart or formal AP."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/sheng/paper6')
PREFIX = 'stereo-epipolar-native-sanity-001'


def main():
    os.chdir(ROOT)
    target = ROOT / 'data/runs' / (PREFIX + '-launch.json')
    log_path = ROOT / 'logs' / (PREFIX + '-controller.log')
    assert not target.exists() and not log_path.exists()
    assert not (ROOT / 'data/runs' / (PREFIX + '-cycle.json')).exists()
    for name in ('F9-coupling-sheng-CPU-001.json', 'F9-training-sheng-CPU-001.json'):
        assert json.loads((ROOT / 'data/engineering' / name).read_text())['state'] == 'passed'
    command = ['bash', '-c', 'source scripts/sheng_env.sh; export PYTHONDONTWRITEBYTECODE=1; exec python experiments/geometry-link/F9/code/engineering_cycle.py --prefix ' + PREFIX]
    with log_path.open('x') as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(state='launched_once_engineering_only', pid=process.pid, launched_unix=time.time(), command=command,
                  prefix=PREFIX, controller_log=str(log_path), formal_AP_authorized_by_this_launcher=False,
                  launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with target.open('x') as stream: json.dump(record, stream, indent=2)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
