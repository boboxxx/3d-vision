"""Start the unique prelocked48-frame evaluation integration exactly once."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/sheng/paper6')
PREFIX = 'stereo-epipolar-evaluation-integration-001'


def main():
    output = ROOT / 'data/runs' / (PREFIX + '-launch.json')
    log = ROOT / 'logs' / (PREFIX + '-controller.log')
    assert not output.exists() and not log.exists()
    assert not (ROOT / 'data/runs' / (PREFIX + '-cycle.json')).exists()
    local = json.loads((ROOT / 'data/engineering/F9-evaluation-local-CPU-005.json').read_text())
    server = json.loads((ROOT / 'data/engineering/F9-evaluation-sheng-CPU-001.json').read_text())
    assert local['state'] == server['state'] == 'passed' and local['tests'] == server['tests'] == 10
    assert local['evaluation_source_identity'] == server['evaluation_source_identity']
    command = ['bash', '-c', 'source scripts/sheng_env.sh; export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled; exec python experiments/geometry-link/F9/evaluation/integration_cycle.py --prefix ' + PREFIX]
    with log.open('x') as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    value = dict(state='launched_once_engineering_no_AP', pid=process.pid, prefix=PREFIX, launched_unix=time.time(),
                 command=command, controller_log=str(log), frames_locked=48, references_locked=8,
                 launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with output.open('x') as stream: json.dump(value, stream, indent=2)
    print(json.dumps(value), flush=True)


if __name__ == '__main__': main()
