"""Unique detached CPU-only neural bridge launcher; never repeats packet PHY."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/sheng/paper6')
PREFIX = 'ecsic-neural-reception-native-001'


def main():
    output = ROOT / 'data/runs' / (PREFIX + '-launch.json')
    log = ROOT / 'logs' / (PREFIX + '-controller.log')
    assert not output.exists() and not log.exists() and not (ROOT / 'data/runs' / (PREFIX + '-cycle.json')).exists()
    local = json.loads((ROOT / 'data/engineering/ecsic-neural-local-CPU-003.json').read_text())
    server = json.loads((ROOT / 'data/engineering/ecsic-neural-sheng-CPU-001.json').read_text())
    assert local['state'] == server['state'] == 'passed' and local['tests'] == server['tests'] == 5 and local['sources'] == server['sources']
    command = ['bash', '-c', 'source scripts/sheng_env.sh; export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1; exec python data/provenance/run-ecsic-neural-cycle-001.py']
    with log.open('x') as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(state='launched_once_CPU_actual_received_neural_crop', pid=process.pid, launched_unix=time.time(),
                  prefix=PREFIX, command=command, controller_log=str(log), received=11, erased=9,
                  launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with output.open('x') as stream: json.dump(record, stream, indent=2)
    print(json.dumps(record), flush=True)


if __name__ == '__main__': main()
