"""Launch the unique CPU-only training JPEG calibration once."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'original-jpeg-rate-calibration-001'


def main():
    output = ROOT / 'data/runs' / (PREFIX + '-launch.json')
    log = ROOT / 'logs' / (PREFIX + '.log')
    assert not output.exists() and not log.exists() and not (ROOT / 'data/runs' / (PREFIX + '.json')).exists()
    assert not (Path('/mnt/d/paper6/runs') / PREFIX).exists()
    command = ['bash', '-c', 'source scripts/sheng_env.sh; export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1; exec nice -n 10 python experiments/original-paper-scenarios/jpeg-rate-calibration-code/calibrate.py']
    with log.open('x') as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    result = dict(state='launched_once_training_only_CPU_calibration', pid=process.pid, launched_unix=time.time(), command=command,
                  launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), log_path=str(log))
    with output.open('x') as stream: json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
