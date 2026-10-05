"""Route a frozen complete endpoint only after actual two-host training closure."""
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
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def accounting(job):
    assert str(job).isdigit()
    raw = subprocess.check_output(['sacct', '-j', str(job), '-n', '-P', '--format=JobIDRaw,State,ExitCode'], text=True)
    rows = [line.split('|') for line in raw.splitlines()]
    assert all([v, 'COMPLETED', '0:0'] in rows for v in [str(job), str(job)+'.0']), raw


def admission(index):
    lock = json.loads((HERE/'inputs.json').read_text())
    for rel, expected in lock['files'].items():
        assert sha(ROOT/rel) == expected, rel
    plan = json.loads((ROOT/'experiments/cost-field/validation-plan-008.json').read_text())
    endpoints = plan['endpoints']
    assert len(endpoints) == 84
    expected = {(17, arm, channel, snr) for arm in ['G','P','S','B'] for channel, snrs in
                [('identity',[10]), ('awgn',range(6,19)), ('rayleigh',range(6,19,2))] for snr in snrs}
    actual = [(e['seed'], e['arm'], e['channel'], e['snr_db']) for e in endpoints]
    assert len(set(actual)) == 84 and set(actual) == expected
    assert all(e['primary'] == (i < 4) for i,e in enumerate(endpoints))
    assert {actual[i] for i in range(4)} == {(17,a,'awgn',10) for a in ['G','P','S','B']}
    assert 0 <= index < 84
    gatepath = ROOT/'data/provenance/cost-field-full-native-local-closure-012.json'
    gate = json.loads(gatepath.read_text())
    assert gate['state'] == 'closed_seed17_all44544_native_local007_and_actual_terminal'
    assert gate['training_raw_job'] == '11424939' and gate['native_audit_job'] == '11425240'
    assert gate['local_stream_pipefail_exit_code'] == 0
    for job in [gate['training_raw_job'], gate['native_audit_job'], gate['export_job']]:
        accounting(job)
    trainingpath = ROOT/'data/runs/cost-field-full-seed17-004/report.json'
    training = json.loads(trainingpath.read_text())
    assert training['seed'] == 17 and training['update_count'] == 44544 and len(training['completed_epochs']) == 12
    assert training['state'] == 'finished_all44544_full_split_task_updates_pending_actual_terminal_and_complete_proof'
    assert gate['training_report_sha256'] == sha(trainingpath)
    proofs = []
    for role in ['native', 'local']:
        rel = f'data/provenance/cost-field-full-seed17-004-{role}-audit-007.json'
        p = ROOT/rel
        assert gate['proof_files'][role] == {'path': rel, 'sha256': sha(p)}
        r = json.loads(p.read_text())
        assert r['state'] == 'passed_complete_training_stream' and r['whole_training_closed']
        assert r['seed'] == 17 and r['checked_updates'] == 44544 and r['unique_frames'] == 3712
        assert r['paired_physical_steps'] == 11136 and r['paired_physical_comparisons'] == 33408
        assert r['report_sha256'] == sha(trainingpath) and r['updates_jsonl_sha256'] == training['updates_jsonl_sha256']
        assert r['actual_terminal']['raw_job_id'] == '11424939' and r['job_id'] == '11424939'
        assert r['verifier_sha256'] == sha(ROOT/'data/provenance/audit-cost-field-full-007.py')
        assert r['protocol_sha256'] == sha(ROOT/'experiments/cost-field/full-proof-protocol-007.md')
        proofs.append(r)
    assert proofs[0]['checked_record_ledger_sha256'] == proofs[1]['checked_record_ledger_sha256'] == gate['checked_record_ledger_sha256']
    assert all(proofs[0][k] == proofs[1][k] for k in ['complete_saved_array_values','gradient_values','complex_uses','paired_physical_steps','paired_physical_comparisons'])
    endpoint = endpoints[index]
    out = ROOT/f"data/runs/cost-field-val-seed17-{endpoint['arm']}-{endpoint['channel']}-{endpoint['snr_db']}-008"
    assert not out.exists(), 'Exclusive endpoint exists; inspect existing result'
    return endpoint, training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True)
    args = parser.parse_args()
    endpoint, training = admission(args.index)
    # No CUDA initialization or output directory precedes the full scientific gate.
    import torch
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability(0) == (12,0)
    assert 'RTX PRO 6000' in torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info()
    assert free >= training['peak_reserved_bytes'] + 2*1024**3, (free,training['peak_reserved_bytes'])
    command = [sys.executable, str(ROOT/'experiments/cost-field/validation_code_008/infer_GPU.py'),
               '--seed','17','--arm',endpoint['arm'],'--channel',endpoint['channel'],'--snr',str(endpoint['snr_db']),
               '--native-training-audit',str(ROOT/'data/provenance/cost-field-full-seed17-004-native-audit-007.json'),
               '--local-training-audit',str(ROOT/'data/provenance/cost-field-full-seed17-004-local-audit-007.json')]
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
