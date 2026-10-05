#!/usr/bin/env bash
#SBATCH --job-name=paper6-cost-full-audit
#SBATCH --account=engdes
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/nfs2/engdes/wc296/paper6/logs/cost-full-audit011-%j.out
#SBATCH --error=/mnt/nfs2/engdes/wc296/paper6/logs/cost-full-audit011-%j.err
set -euo pipefail
cd /mnt/nfs2/engdes/wc296/paper6
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
python3 - <<'PY'
import hashlib,json,subprocess,time
from pathlib import Path
lock=json.loads(Path('experiments/cost-field/full-audit-code-011/inputs.json').read_text())
assert lock['seed']==17 and lock['training_raw_job']=='11424939' and lock['updates']==44544
for path,expected in lock['files'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
output=Path(lock['native_output'])
assert not output.exists() and not output.with_suffix('.records.jsonl').exists(),'Exclusive output exists; inspect original run'
for attempt in range(6):
    rows=[line.split('|') for line in subprocess.check_output(['sacct','-j','11424939','--noheader','--parsable2','--format=JobIDRaw,State,ExitCode'],text=True).splitlines()]
    if all([key,'COMPLETED','0:0'] in rows for key in ['11424939','11424939.0']):break
    if attempt<5:time.sleep(2)
else:raise RuntimeError('Parent rawjob and .0 normal completion not established; retain failure')
r=json.loads(Path('data/runs/cost-field-full-seed17-004/report.json').read_text())
assert r['seed']==17 and r['job_id']=='11424939' and r['update_count']==44544 and len(r['completed_epochs'])==12
assert r['state']=='finished_all44544_full_split_task_updates_pending_actual_terminal_and_complete_proof'
PY
srun --ntasks=1 envs/torch-cu128-001/bin/python data/provenance/audit-cost-field-full-007.py --mode native --seed 17 --count 44544 --output data/provenance/cost-field-full-seed17-004-native-audit-007.json
