"""Actual terminal-bound full003 stream, no retained checkpoint collection."""
import argparse
import importlib.util
import json
import subprocess
from pathlib import Path
from native import ROOT, HERE, LAMBDAS, sha, selected

parser = argparse.ArgumentParser()
parser.add_argument('--candidate', type=int, choices=[4, 5], required=True)
parser.add_argument('--native-array', required=True)
args = parser.parse_args()
assert args.native_array.isdigit()
lock, directory, report = selected(args.candidate)
fact = json.loads((ROOT/'data/provenance'/f'ecsic-candidate{args.candidate}-native-execution-013.json').read_text())
assert fact['state'] == 'whole_native_auditor_exit0_pending_actual_native_terminal' and fact['auditor_exit_code'] == 0
assert fact['candidate'] == args.candidate and fact['native_array_job'] == args.native_array
assert fact['training_actual_raw_job'] == report['job_id'] and fact['training_report_sha256'] == sha(directory/'report.json')
assert fact['inputs_sha256'] == sha(HERE/'inputs.json')
job = fact['native_actual_raw_job']
raw = subprocess.check_output(['sacct', '-j', args.native_array+'_'+str(args.candidate), '-n', '-P',
    '--format=JobIDRaw,State,ExitCode'], text=True)
rows = [line.split('|') for line in raw.splitlines()]
assert all([v, 'COMPLETED', '0:0'] in rows for v in [job, job+'.0']), raw
prefix = f'ecsic-KITTI-lambda{LAMBDAS[args.candidate]:.3g}-seed17-002'
p = ROOT/'data/provenance'/f'{prefix}-native-audit-003.json'
proof = json.loads(p.read_text())
assert sha(p) == fact['proof_sha256']
assert proof['candidate'] == args.candidate and proof['seed'] == 17
assert proof['state'] == 'passed_whole37120_source_training_records_and_all10_epoch_states'
assert proof['checked_updates'] == 37120 and proof['checked_epoch_checkpoints'] == 10
assert proof['metadata']['report_sha256'] == sha(directory/'report.json')
assert proof['verifier_sha256'] == sha(ROOT/'data/provenance/audit-ecsic-KITTI-formal-003.py')
assert proof['count_repair_protocol_sha256'] == sha(ROOT/'experiments/original-paper-scenarios/ecsic-KITTI-formal-audit-alias-repair-003.md')
assert proof['protocol_sha256'] == sha(ROOT/'experiments/original-paper-scenarios/ecsic-KITTI-adaptation-protocol-002.md')
spec = importlib.util.spec_from_file_location('formal003', ROOT/'data/provenance/audit-ecsic-KITTI-formal-003.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.exported(args.candidate)
