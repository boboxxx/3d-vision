"""Admit unchanged fresh public30 source stage1 after actual engineering proof."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


lock = json.loads((HERE/'inputs.json').read_text())
for rel, expected in lock['files'].items():
    assert sha(ROOT/rel) == expected, rel
gatepath = ROOT/'data/provenance/original-public30-engineering-native-local-closure-006.json'
gate = json.loads(gatepath.read_text())
assert gate['state'] == 'closed_actual_GPU_native_local_original_public_engineering003'
assert gate['input_sha256'] == sha(ROOT/'experiments/original-paper-scenarios/original-public-train-code-003/inputs.json')
assert gate['local_auditor_actual_exit_code'] == 0
assert gate['GPU_engineering_job'] == '11425104' and gate['native_audit_job'] == '11425880'
for job in [gate['GPU_engineering_job'],gate['native_audit_job']]:
    raw = subprocess.check_output(['sacct','-j',job,'-n','-P','--format=JobIDRaw,State,ExitCode'],text=True)
    rows = [line.split('|') for line in raw.splitlines()]
    assert all([key,'COMPLETED','0:0'] in rows for key in [job,job+'.0']), raw
directory = ROOT/'data/runs/original-public30-engineering-seed17-003-stage1'
r = json.loads((directory/'report.json').read_text())
assert r['scope'] == 'engineering' and r['stage'] == 1 and r['seed'] == 17 and r['update_count'] == 3
assert r['job_id'] == '11425104' and len(r['completed_epochs']) == 1
assert r['state'] == 'finished_complete_public_semantic_stage_pending_actual_terminal_and_audit'
assert gate['report_sha256'] == sha(directory/'report.json')
assert gate['initial_sha256'] == r['initial_sha256'] == sha(directory/'initial.pt')
assert gate['records_sha256'] == r['records_sha256'] == sha(directory/'updates.jsonl')
terminalpath = ROOT/'data/provenance/original-public30-engineering-GPU-terminal-004.json'
assert gate['GPU_terminal_sha256'] == sha(terminalpath)
proofs = []
for role in ['native','local']:
    rel = f'data/provenance/original-public30-engineering-{role}-audit-004.json'
    p = ROOT/rel
    assert gate['proof_files'][role] == {'path':rel,'sha256':sha(p)}
    v = json.loads(p.read_text())
    assert v['state'] == 'passed_complete_original_public_semantic_stage' and v['actual_terminal_closed']
    assert v['scope'] == 'engineering' and v['stage'] == 1 and v['updates'] == 3 and v['mode'] == role
    assert v['report_sha256'] == gate['report_sha256'] and v['records_sha256'] == gate['records_sha256']
    assert v['terminal_sha256'] == gate['GPU_terminal_sha256'] and v['input_sha256'] == gate['input_sha256']
    assert v['verifier_sha256'] == sha(ROOT/'experiments/original-paper-scenarios/original-public-train-code-003/audit.py')
    proofs.append(v)
assert proofs[0]['epochs'] == proofs[1]['epochs'] and len(proofs[0]['epochs']) == 1
assert proofs[1]['native_audit_sha256'] == gate['proof_files']['native']['sha256']
assert not (ROOT/'data/runs/original-public30-formal-seed17-003-stage1').exists()
command = [sys.executable,str(ROOT/'experiments/original-paper-scenarios/original-public-train-code-003/train.py'),
           '--scope','formal','--stage','1','--gate',str(gatepath)]
os.execv(sys.executable,command)
