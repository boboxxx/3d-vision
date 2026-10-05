"""Bind unchanged original engineering audit003 to actual GPU terminal rows."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


lock = json.loads((HERE/'inputs.json').read_text())
for rel, expected in lock['files'].items():
    assert sha(ROOT/rel) == expected, rel
directory = ROOT/'data/runs/original-public30-engineering-seed17-003-stage1'
r = json.loads((directory/'report.json').read_text())
assert r['state'] == 'finished_complete_public_semantic_stage_pending_actual_terminal_and_audit'
assert r['scope'] == 'engineering' and r['stage'] == 1 and r['seed'] == 17
assert r['job_id'] == '11425104' and r['update_count'] == 3 and len(r['completed_epochs']) == 1
assert r['input_sha256'] == sha(ROOT/'experiments/original-paper-scenarios/original-public-train-code-003/inputs.json')
terminal = ROOT/'data/provenance/original-public30-engineering-GPU-terminal-004.json'
output = ROOT/'data/provenance/original-public30-engineering-native-audit-004.json'
assert not terminal.exists() and not output.exists()
for attempt in range(6):
    raw = subprocess.check_output(['sacct','-j','11425104','-n','-P','--format=JobIDRaw,State,ExitCode'], text=True)
    rows = {v[0]:{'State':v[1],'ExitCode':v[2]} for v in [line.split('|') for line in raw.splitlines()]}
    if all(rows.get(v) == {'State':'COMPLETED','ExitCode':'0:0'} for v in ['11425104','11425104.0']):
        break
    if attempt < 5:
        time.sleep(2)
else:
    raise RuntimeError(raw)
with terminal.open('x') as f:
    json.dump({'job_id':'11425104','actual_job_and_srun_step_completed_exit0':True,
               'rows':rows,'raw_sacct':raw,'checked_unix':time.time()},f,indent=2)
command = [sys.executable,str(ROOT/'experiments/original-paper-scenarios/original-public-train-code-003/audit.py'),
           '--directory',str(directory),'--terminal',str(terminal),'--output',str(output)]
os.execv(sys.executable,command)
