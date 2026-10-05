"""Admit unchanged source stage2 after actual complete stage1 two-host closure."""
import os
import subprocess
import sys
from pathlib import Path
from close_stage1 import ROOT, HERE, JOBS, RUN, NATIVE, LOCAL, TERMINAL, GATE, sha, read, locks, report_check, proof_check

locks()
gate = read(ROOT/GATE)
assert gate['state'] == 'closed_original_public30_stage1_all44544_all12_native_local_actual_normal_terminals'
assert gate['stage'] == 1 and gate['seed'] == 17 and gate['updates'] == 44544 and gate['epoch_checkpoints'] == 12
assert gate['closure_inputs_sha256'] == sha(HERE/'inputs.json') and gate['jobs'] == JOBS
assert gate['local_auditor_actual_exit_code'] == gate['export_actual_exit_code'] == 0
for job in JOBS:
    raw = subprocess.check_output(['sacct', '-j', job, '-n', '-P', '--format=JobIDRaw,State,ExitCode'], text=True)
    rows = [line.split('|') for line in raw.splitlines()]
    assert all([v, 'COMPLETED', '0:0'] in rows for v in [job, job+'.0']), raw
directory = ROOT/RUN; report = read(directory/'report.json'); report_check(report)
assert gate['report_sha256'] == sha(directory/'report.json')
assert gate['records_sha256'] == report['records_sha256'] == sha(directory/'updates.jsonl')
proofs = []
for role, rel in [('native', NATIVE), ('local', LOCAL)]:
    path = ROOT/rel; assert gate['proof_files'][role] == dict(path=rel, sha256=sha(path))
    proof = read(path); proof_check(proof, role, gate['report_sha256'], gate['records_sha256'], sha(ROOT/TERMINAL))
    proofs.append(proof)
assert proofs[0]['epochs'] == proofs[1]['epochs']
assert proofs[1]['native_audit_sha256'] == sha(ROOT/NATIVE)
for name in ['initial.pt', 'checkpoint-epoch12.pt']:
    p = directory/name; saved = gate['native_backup_files'][name]
    assert p.stat().st_size == saved['bytes'] and sha(p) == saved['sha256']
assert not (ROOT/'data/runs/original-public30-formal-seed17-003-stage2').exists()
command = [sys.executable, str(ROOT/'experiments/original-paper-scenarios/original-public-train-code-003/train.py'),
           '--scope', 'formal', '--stage', '2', '--predecessor-report', str(directory/'report.json'),
           '--predecessor-audit', str(ROOT/NATIVE), '--gate',
           str(ROOT/'data/provenance/original-public30-engineering-native-local-closure-006.json')]
os.execv(sys.executable, command)
