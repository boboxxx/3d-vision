"""Export unchanged full003 bytes after actual candidate3 native audit closure."""
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


lock = json.loads((HERE/'inputs.json').read_text())
for f, expected in lock['files'].items():
    assert sha(ROOT/f) == expected, f
for job in ['11426641', '11426848']:
    raw = subprocess.check_output(['sacct','-j',job,'-n','-P','--format=JobIDRaw,State,ExitCode'], text=True)
    rows = [line.split('|') for line in raw.splitlines()]
    assert all([v,'COMPLETED','0:0'] in rows for v in [job,job+'.0']), raw
p = ROOT/'data/provenance/ecsic-KITTI-lambda0.03-seed17-002-native-audit-003.json'
r = json.loads(p.read_text())
assert r['state'] == 'passed_whole37120_source_training_records_and_all10_epoch_states'
assert r['candidate'] == 3 and r['seed'] == 17 and r['checked_updates'] == 37120 and r['checked_epoch_checkpoints'] == 10
training = ROOT/'data/runs/ecsic-KITTI-lambda0.03-seed17-002/report.json'
assert r['metadata']['report_sha256'] == sha(training)
assert r['verifier_sha256'] == sha(ROOT/'data/provenance/audit-ecsic-KITTI-formal-003.py')
assert r['count_repair_protocol_sha256'] == sha(ROOT/'experiments/original-paper-scenarios/ecsic-KITTI-formal-audit-alias-repair-003.md')
assert r['protocol_sha256'] == sha(ROOT/'experiments/original-paper-scenarios/ecsic-KITTI-adaptation-protocol-002.md')
spec = importlib.util.spec_from_file_location('formal003', ROOT/'data/provenance/audit-ecsic-KITTI-formal-003.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.exported(3)
