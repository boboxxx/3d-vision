"""Gate a whole unchanged007 export on actual native training closure."""
import hashlib
import importlib.util
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
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def terminal(job):
    for attempt in range(6):
        raw = subprocess.check_output(['sacct','-j',job,'-n','-P','--format=JobIDRaw,State,ExitCode'], text=True)
        rows = [line.split('|') for line in raw.splitlines()]
        if all([v,'COMPLETED','0:0'] in rows for v in [job,job+'.0']):
            return
        if attempt < 5:
            time.sleep(2)
    raise RuntimeError(raw)


lock = json.loads((HERE/'inputs.json').read_text())
assert lock['seed'] == 17 and lock['count'] == 44544
assert lock['training_raw_job'] == '11424939' and lock['native_audit_job'] == '11425240'
for rel, expected in lock['files'].items():
    assert sha(ROOT/rel) == expected, rel
for job in [lock['training_raw_job'], lock['native_audit_job']]:
    terminal(job)
p = ROOT/'data/provenance/cost-field-full-seed17-004-native-audit-007.json'
r = json.loads(p.read_text())
assert r['state'] == 'passed_complete_training_stream' and r['whole_training_closed']
assert r['seed'] == 17 and r['job_id'] == '11424939'
assert r['requested_updates'] == r['checked_updates'] == 44544 and r['unique_frames'] == 3712
assert r['paired_physical_steps'] == 11136 and r['paired_physical_comparisons'] == 33408
assert r['checked_record_ledger_sha256'] == sha(p.with_suffix('.records.jsonl'))
training = ROOT/'data/runs/cost-field-full-seed17-004/report.json'
t = json.loads(training.read_text())
assert t['update_count'] == 44544 and len(t['completed_epochs']) == 12
assert t['state'] == 'finished_all44544_full_split_task_updates_pending_actual_terminal_and_complete_proof'
assert r['report_sha256'] == sha(training)
assert r['updates_jsonl_sha256'] == t['updates_jsonl_sha256']
assert r['verifier_sha256'] == sha(ROOT/'data/provenance/audit-cost-field-full-007.py')
assert r['protocol_sha256'] == sha(ROOT/'experiments/cost-field/full-proof-protocol-007.md')
assert r['actual_terminal']['raw_job_id'] == '11424939'
print(json.dumps({'actual_export_job':os.environ['SLURM_JOB_ID'],'native_proof_sha256':sha(p)}),file=sys.stderr,flush=True)
spec = importlib.util.spec_from_file_location('full007', ROOT/'data/provenance/audit-cost-field-full-007.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.exported(17,44544)
