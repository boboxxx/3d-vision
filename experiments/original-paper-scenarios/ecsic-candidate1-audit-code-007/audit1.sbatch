#!/usr/bin/env bash
#SBATCH --job-name=paper6-ecsic-c1-audit
#SBATCH --account=engdes
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/nfs2/engdes/wc296/paper6/logs/ecsic-candidate1-audit007-%j.out
#SBATCH --error=/mnt/nfs2/engdes/wc296/paper6/logs/ecsic-candidate1-audit007-%j.err
set -euo pipefail
cd /mnt/nfs2/engdes/wc296/paper6
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
python3 - <<'PY'
import hashlib,json,subprocess,time
from pathlib import Path
lock=json.loads(Path('experiments/original-paper-scenarios/ecsic-candidate1-audit-code-007/inputs.json').read_text())
assert lock['candidate']==1 and lock['training_raw_job']=='11425119'
for path,expected in lock['files'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
assert not Path(lock['native_output']).exists(),'Exclusive output already exists; inspect original audit'
for attempt in range(6):
    rows=[line.split('|') for line in subprocess.check_output(['sacct','-j','11425119','--noheader','--parsable2','--format=JobIDRaw,State,ExitCode'],text=True).splitlines()]
    if all([key,'COMPLETED','0:0'] in rows for key in ['11425119','11425119.0']):break
    if attempt<5:time.sleep(2)
else:raise RuntimeError('Parent rawjob and .0 normal completion not established; retain failure')
r=json.loads(Path('data/runs/ecsic-KITTI-lambda0.003-seed17-002/report.json').read_text())
assert r['candidate']==1 and r['job_id']=='11425119' and r['update_count']==37120
PY
srun --ntasks=1 envs/torch-cu128-001/bin/python data/provenance/audit-ecsic-KITTI-formal-003.py --mode native --candidate 1 --output data/provenance/ecsic-KITTI-lambda0.003-seed17-002-native-audit-003.json
