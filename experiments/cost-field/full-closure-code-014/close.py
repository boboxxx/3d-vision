"""Materialize admission012 only from actual whole007 two-host closures."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
JOBS = ['11424939','11425240','11425814']
NATIVE = 'data/provenance/cost-field-full-seed17-004-native-audit-007.json'
LOCAL = 'data/provenance/cost-field-full-seed17-004-local-audit-007.json'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def locks():
    lock = json.loads((HERE/'inputs.json').read_text())
    assert lock['actual_jobs'] == JOBS
    for rel, expected in lock['files'].items():
        assert sha(ROOT/rel) == expected, rel
    return lock


def proof(r, report_sha, updates_sha):
    assert r['state'] == 'passed_complete_training_stream' and r['whole_training_closed']
    assert r['job_id'] == '11424939' and r['seed'] == 17
    assert r['requested_updates'] == r['checked_updates'] == 44544 and r['unique_frames'] == 3712
    assert r['paired_physical_steps'] == 11136 and r['paired_physical_comparisons'] == 33408
    assert r['report_sha256'] == report_sha and r['updates_jsonl_sha256'] == updates_sha
    assert r['actual_terminal']['raw_job_id'] == '11424939'
    assert r['verifier_sha256'] == sha(ROOT/'data/provenance/audit-cost-field-full-007.py')
    assert r['protocol_sha256'] == sha(ROOT/'experiments/cost-field/full-proof-protocol-007.md')


def collect():
    locks()
    accounting = {}
    for job in JOBS:
        raw = subprocess.check_output(['sacct','-j',job,'-n','-P','--format=JobIDRaw,State,ExitCode'],text=True)
        rows = [line.split('|') for line in raw.splitlines()]
        assert all([key,'COMPLETED','0:0'] in rows for key in [job,job+'.0']), raw
        accounting[job] = raw
    p = ROOT/'data/runs/cost-field-full-seed17-004/report.json'
    t = json.loads(p.read_text())
    assert t['seed'] == 17 and t['job_id'] == '11424939' and t['update_count'] == 44544
    assert len(t['completed_epochs']) == 12
    assert t['state'] == 'finished_all44544_full_split_task_updates_pending_actual_terminal_and_complete_proof'
    report_sha = sha(p)
    updates_sha = sha(p.parent/'updates.jsonl')
    assert updates_sha == t['updates_jsonl_sha256']
    p = ROOT/NATIVE
    b = p.read_bytes()
    n = json.loads(b)
    proof(n,report_sha,updates_sha)
    assert sha(p.with_suffix('.records.jsonl')) == n['checked_record_ledger_sha256']
    print(json.dumps({'inputs_sha256':sha(HERE/'inputs.json'),'accounting':accounting,
                      'training_report_sha256':report_sha,'updates_jsonl_sha256':updates_sha,
                      'native_proof_utf8':b.decode('utf-8'),'native_proof_sha256':sha(p),
                      'checked_unix':time.time()}))


def close():
    locks()
    out = ROOT/'data/provenance/cost-field-full-native-local-closure-012.json'
    assert not out.exists(), 'Exclusive runtime gate exists; inspect prior closure'
    launcher = json.loads((ROOT/'data/provenance/cost-field-full-local-stream-launch-013.json').read_text())
    assert launcher['state'] == 'local_pipe_exit0_pending_actual_export_terminal_and_two_host_closure'
    assert launcher['inputs_sha256'] == sha(ROOT/'experiments/cost-field/full-stream-code-013/inputs.json')
    assert launcher['local_reader_exit_code'] == launcher['ssh_export_exit_code'] == launcher['pipefail_exit_code'] == 0
    lpath = ROOT/LOCAL
    l = json.loads(lpath.read_text())
    assert sha(lpath.with_suffix('.records.jsonl')) == l['checked_record_ledger_sha256']
    remote = ('cd /mnt/nfs2/engdes/wc296/paper6 && PYTHONDONTWRITEBYTECODE=1 '
              'envs/torch-cu128-001/bin/python experiments/cost-field/full-closure-code-014/close.py --collect')
    raw = subprocess.check_output(['ssh','artemis-ood',remote],text=True)
    c = json.loads(raw)
    assert c['inputs_sha256'] == sha(HERE/'inputs.json')
    b = c['native_proof_utf8'].encode('utf-8')
    assert hashlib.sha256(b).hexdigest() == c['native_proof_sha256']
    n = json.loads(b)
    for r in [n,l]:
        proof(r,c['training_report_sha256'],c['updates_jsonl_sha256'])
    keys = ['checked_record_ledger_sha256','complete_saved_array_values','gradient_values',
            'complex_uses','paired_physical_steps','paired_physical_comparisons']
    assert all(n[k] == l[k] for k in keys)
    assert all(n[k] > 0 for k in ['complete_saved_array_values','gradient_values','complex_uses'])
    npath = ROOT/NATIVE
    if npath.exists():
        assert npath.read_bytes() == b
    else:
        with npath.open('xb') as f:
            f.write(b)
    gate = {'state':'closed_seed17_all44544_native_local007_and_actual_terminal',
            'training_raw_job':JOBS[0],'native_audit_job':JOBS[1],'export_job':JOBS[2],
            'local_stream_pipefail_exit_code':0,'training_report_sha256':c['training_report_sha256'],
            'proof_files':{role:{'path':rel,'sha256':sha(ROOT/rel)} for role,rel in [('native',NATIVE),('local',LOCAL)]},
            'checked_record_ledger_sha256':n['checked_record_ledger_sha256'],
            'actual_accounting':c['accounting'],'checked_unix':time.time(),
            'closure_execution_inputs_sha256':sha(HERE/'inputs.json'),
            'AP_or_detection_advantage_claimed':False,'limitation':n['limitation']}
    with out.open('x') as f:
        json.dump(gate,f,indent=2)
        f.write('\n')
    print(json.dumps({'state':gate['state'],'gate_sha256':sha(out)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--collect',action='store_true')
    group.add_argument('--close',action='store_true')
    args = parser.parse_args()
    collect() if args.collect else close()
