"""Unchanged whole003 audit for the two final, actually selected GPU tasks."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
LAMBDAS = {4: .1, 5: .3}


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def locked():
    lock = json.loads((HERE/'inputs.json').read_text())
    assert all(sha(ROOT/p) == value for p, value in lock['files'].items())
    return lock


def selected(candidate):
    lock = locked()
    assert candidate in LAMBDAS
    directory = ROOT/'data/runs'/f'ecsic-KITTI-lambda{LAMBDAS[candidate]:.3g}-seed17-002'
    r = json.loads((directory/'report.json').read_text())
    assert r['candidate'] == candidate and int(r['array_task_id']) == candidate
    assert r['array_job_id'] == lock['training_array']
    assert r['lambda_RD'] == LAMBDAS[candidate] and r['seed'] == 17
    assert r['state'] == 'finished_all37120_source_RD_updates_pending_actual_terminal_full_audit_and_real_byte_calibration'
    assert r['update_count'] == 37120 and len(r['completed_epochs']) == 10
    job = r['job_id']
    raw = subprocess.check_output(['sacct', '-j', lock['training_array']+'_'+str(candidate), '-n', '-P',
        '--format=JobIDRaw,State,ExitCode'], text=True)
    rows = [line.split('|') for line in raw.splitlines()]
    assert all([v, 'COMPLETED', '0:0'] in rows for v in [job, job+'.0']), raw
    return lock, directory, r


def main():
    candidate = int(os.environ['SLURM_ARRAY_TASK_ID'])
    lock, directory, report = selected(candidate)
    prefix = f'ecsic-KITTI-lambda{LAMBDAS[candidate]:.3g}-seed17-002'
    output = ROOT/'data/provenance'/f'{prefix}-native-audit-003.json'
    path = ROOT/'data/provenance'/f'ecsic-candidate{candidate}-native-execution-013.json'
    assert not output.exists() and not path.exists()
    fact = dict(state='starting_whole_native_audit_not_passed', candidate=candidate,
        training_actual_raw_job=report['job_id'], training_array_selector=lock['training_array']+'_'+str(candidate),
        native_actual_raw_job=os.environ['SLURM_JOB_ID'], native_array_job=os.environ['SLURM_ARRAY_JOB_ID'],
        inputs_sha256=sha(HERE/'inputs.json'), training_report_sha256=sha(directory/'report.json'))
    path.write_text(json.dumps(fact, indent=2)+'\n')
    command = [sys.executable, str(ROOT/'data/provenance/audit-ecsic-KITTI-formal-003.py'),
        '--mode', 'native', '--candidate', str(candidate), '--output', str(output)]
    code = subprocess.run(command).returncode
    fact['auditor_exit_code'] = code
    if code == 0:
        proof = json.loads(output.read_text())
        assert proof['state'] == 'passed_whole37120_source_training_records_and_all10_epoch_states'
        assert proof['candidate'] == candidate and proof['checked_updates'] == 37120 and proof['checked_epoch_checkpoints'] == 10
        fact.update(state='whole_native_auditor_exit0_pending_actual_native_terminal', proof_sha256=sha(output))
    else:
        fact['state'] = 'failed_retained'
    path.write_text(json.dumps(fact, indent=2)+'\n')
    sys.exit(code)


if __name__ == '__main__':
    main()
