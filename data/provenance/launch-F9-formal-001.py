"""Unique final-only F9 formal cycle launcher; no restart on observation timeout."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/sheng/paper6')
PREFIX = 'stereo-epipolar-native-seed17-001'


def main():
    output = ROOT / 'data/runs' / (PREFIX + '-launch.json')
    log = ROOT / 'logs' / (PREFIX + '-controller.log')
    assert not output.exists() and not log.exists()
    assert not (ROOT / 'data/runs' / (PREFIX + '-cycle.json')).exists()
    local = json.loads((ROOT / 'data/engineering/F9-evaluation-local-CPU-007.json').read_text())
    server = json.loads((ROOT / 'data/engineering/F9-evaluation-sheng-CPU-002.json').read_text())
    assert local['state'] == server['state'] == 'passed' and local['tests'] == server['tests'] == 12
    assert local['evaluation_source_identity'] == server['evaluation_source_identity']
    integration = json.loads((ROOT / 'data/provenance/stereo-epipolar-evaluation-integration-001-closure.json').read_text())
    assert integration['actual_terminal'] and integration['frames'] == 48 and integration['references'] == 8
    command = ['bash', '-c', 'source scripts/sheng_env.sh; export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled; exec python experiments/geometry-link/F9/evaluation/formal_cycle.py --prefix ' + PREFIX]
    with log.open('x') as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    value = dict(state='launched_once_formal_four_arm_final_only', pid=process.pid, prefix=PREFIX,
                 launched_unix=time.time(), command=command, controller_log=str(log),
                 updates_per_arm_locked=3340, final_endpoints_locked=16,
                 launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with output.open('x') as stream: json.dump(value, stream, indent=2)
    print(json.dumps(value), flush=True)


if __name__ == '__main__': main()
