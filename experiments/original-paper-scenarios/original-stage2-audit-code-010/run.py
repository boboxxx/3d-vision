"""Bind unchanged whole source audit003 to actual completed stage2 GPU job."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


lock = json.loads((HERE/'inputs.json').read_text())
assert all(sha(ROOT/p) == v for p, v in lock['files'].items())
directory = ROOT/'data/runs/original-public30-formal-seed17-003-stage2'
r = json.loads((directory/'report.json').read_text())
assert r['state'] == 'finished_complete_public_semantic_stage_pending_actual_terminal_and_audit'
assert r['scope'] == 'formal' and r['stage'] == 2 and r['seed'] == 17
assert r['job_id'] == '11429413' and r['update_count'] == 37120 and len(r['completed_epochs']) == 10
assert r['input_sha256'] == sha(ROOT/'experiments/original-paper-scenarios/original-public-train-code-003/inputs.json')
parent = ROOT/'data/runs/original-public30-formal-seed17-003-stage1/report.json'
previous = ROOT/'data/runs/original-public30-formal-seed17-003-stage1/checkpoint-epoch12.pt'
assert r['predecessor_report_sha256'] == sha(parent)
assert r['predecessor_audit_sha256'] == sha(ROOT/'data/provenance/original-public30-stage1-native-audit-007.json')
assert r['predecessor_checkpoint_sha256'] == sha(previous)
terminal = ROOT/'data/provenance/original-public30-stage2-GPU-terminal-010.json'
output = ROOT/'data/provenance/original-public30-stage2-native-audit-010.json'
assert not terminal.exists() and not output.exists()
for attempt in range(6):
    raw = subprocess.check_output(['sacct', '-j', '11429413', '-n', '-P',
                                  '--format=JobIDRaw,State,ExitCode'], text=True)
    rows = {v[0]: dict(State=v[1], ExitCode=v[2]) for v in [line.split('|') for line in raw.splitlines()]}
    if all(rows.get(v) == dict(State='COMPLETED', ExitCode='0:0') for v in ['11429413', '11429413.0']):
        break
    if attempt < 5: time.sleep(2)
else:
    raise RuntimeError(raw)
with terminal.open('x') as f:
    json.dump(dict(job_id='11429413', actual_job_and_srun_step_completed_exit0=True,
                   rows=rows, raw_sacct=raw, checked_unix=time.time()), f, indent=2)
command = [sys.executable, str(ROOT/'experiments/original-paper-scenarios/original-public-train-code-003/audit.py'),
           '--directory', str(directory), '--terminal', str(terminal), '--output', str(output),
           '--predecessor-checkpoint', str(previous)]
os.execv(sys.executable, command)
