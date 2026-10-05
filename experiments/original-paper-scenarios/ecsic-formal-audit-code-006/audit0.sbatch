#!/usr/bin/env bash
#SBATCH --job-name=paper6-ecsic-full-audit
#SBATCH --account=engdes
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/nfs2/engdes/wc296/paper6/logs/ecsic-full-audit006-%j.out
#SBATCH --error=/mnt/nfs2/engdes/wc296/paper6/logs/ecsic-full-audit006-%j.err
set -euo pipefail
cd /mnt/nfs2/engdes/wc296/paper6
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
python3 - <<'PY'
import hashlib,json,subprocess,time
from pathlib import Path
lock=json.loads(Path('experiments/original-paper-scenarios/ecsic-formal-audit-code-006/inputs.json').read_text())
assert hashlib.sha256(Path('data/provenance/audit-ecsic-KITTI-formal-003.py').read_bytes()).hexdigest()==lock['verifier_sha256']
for attempt in range(6):
 rows=[line.split('|') for line in subprocess.check_output(['sacct','-j','11425048','--noheader','--parsable2','--format=JobIDRaw,State,ExitCode'],text=True).splitlines()]
 if all([key,'COMPLETED','0:0'] in rows for key in ['11425048','11425048.0']):break
 if attempt<5:time.sleep(2)
else:raise RuntimeError('Parent normal rawjob and .0 completion not established; preserve failure')
PY
srun --ntasks=1 envs/torch-cu128-001/bin/python data/provenance/audit-ecsic-KITTI-formal-003.py --mode native --candidate 0 --output data/provenance/ecsic-KITTI-lambda0.001-seed17-002-native-audit-003.json
