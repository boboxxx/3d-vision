"""Actual source008 complete engineering audit after matching array parent."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument('--training-array', required=True)
args = parser.parse_args()
assert args.training_array.isdigit()
rate = int(os.environ['SLURM_ARRAY_TASK_ID'])
assert rate in (10, 50)
lock = json.loads((HERE/'inputs.json').read_text())
assert all(sha(ROOT/p) == value for p, value in lock['files'].items())
directory = ROOT/'data/runs'/f'original-public{rate}-engineering-seed17-008-stage1'
report = json.loads((directory/'report.json').read_text())
assert report['scope'] == 'engineering' and report['stage'] == 1 and report['nominal_rate'] == rate
assert report['update_count'] == 3 and len(report['completed_epochs']) == 1
assert report['state'] == 'finished_complete_public_semantic_stage_pending_actual_terminal_and_audit'
fact = json.loads((ROOT/'data/provenance'/f'original-public{rate}-engineering-GPU-launch-009.json').read_text())
assert fact['training_actual_job'] == report['job_id'] and fact['training_array_job'] == args.training_array
assert fact['training_array_task'] == str(rate) and fact['inputs_sha256'] == sha(HERE/'inputs.json')
raw = subprocess.check_output(['sacct', '-j', args.training_array+'_'+str(rate), '-n', '-P',
    '--format=JobIDRaw,State,ExitCode'], text=True)
rows = {row[0]: dict(State=row[1], ExitCode=row[2]) for row in [line.split('|') for line in raw.splitlines()]}
job = report['job_id']
assert all(rows[key] == dict(State='COMPLETED', ExitCode='0:0') for key in [job, job+'.0']), raw
terminal = ROOT/'data/provenance'/f'original-public{rate}-engineering-GPU-terminal-009.json'
assert not terminal.exists()
terminal.write_text(json.dumps(dict(job_id=job, rows=rows,
    actual_job_and_srun_step_completed_exit0=True, actual_native_audit_job=os.environ['SLURM_JOB_ID']), indent=2)+'\n')
output = ROOT/'data/provenance'/f'original-public{rate}-engineering-native-audit-009.json'
command = [sys.executable, str(ROOT/'experiments/original-paper-scenarios/original-source-rate-code-008/audit.py'),
    '--directory', str(directory), '--terminal', str(terminal), '--output', str(output)]
subprocess.run(command, check=True)
proof = json.loads(output.read_text())
assert proof['state'] == 'passed_complete_original_source_rate_semantic_stage008' and proof['updates'] == 3
