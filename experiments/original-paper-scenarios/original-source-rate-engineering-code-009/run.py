"""Admit three genuine source updates after actual CPU008 two-host proof."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


lock = json.loads((HERE/'inputs.json').read_text())
assert all(sha(ROOT/p) == value for p, value in lock['files'].items())
rate = int(os.environ['SLURM_ARRAY_TASK_ID'])
assert rate in (10, 50)
gate = json.loads((ROOT/lock['CPU_gate']).read_text())
assert gate['state'] == 'closed_actual_two_host_all3_six_source_phases_CPU008'
assert gate['local_actual_exit_code'] == 0 and gate['native_raw_and_dot0_completed_exit0']
assert gate['inputs_sha256'] == sha(ROOT/'experiments/original-paper-scenarios/original-source-rate-code-008/inputs.json')
job = gate['actual_native_CPU_job']
raw = subprocess.check_output(['sacct', '-j', job, '-n', '-P', '--format=JobIDRaw,State,ExitCode'], text=True)
rows = [line.split('|') for line in raw.splitlines()]
assert all([key, 'COMPLETED', '0:0'] in rows for key in [job, job+'.0']), raw
proofs = []
for role in ('local', 'native'):
    item = gate['proof_files'][role]
    p = ROOT/item['path']
    assert sha(p) == item['sha256']
    proof = json.loads(p.read_text())
    assert proof['state'] == 'passed_all3_six_source_phases_complete_gradients_and_frozen30_bit_identity'
    assert proof['inputs_sha256'] == gate['inputs_sha256']
    assert [v['nominal_rate'] for v in proof['rates']] == [30, 10, 50]
    for v in proof['rates']:
        conditions = v['complete_source_loss_gradient_conditions']
        assert [(c['stage'], c['epoch']) for c in conditions] == [(1, 1), (1, 3), (2, 1), (3, 1), (4, 1), (4, 6)]
        assert all(c['states_unchanged'] and c['active_gradient_tensors'] > 0 for c in conditions)
    assert proof['rates'][0]['nominal30_bit_identical_frozen_control']
    proofs.append(proof)
assert proofs[0]['raw_input_sha256'] == proofs[1]['raw_input_sha256'] == gate['raw_input_sha256']
assert proofs[0]['boxes'] == proofs[1]['boxes']
path = ROOT/'data/provenance'/f'original-public{rate}-engineering-GPU-launch-009.json'
assert not path.exists()
fact = dict(state='starting_actual_three_update_GPU_engineering_not_passed', nominal_rate=rate,
    training_actual_job=os.environ['SLURM_JOB_ID'], training_array_job=os.environ['SLURM_ARRAY_JOB_ID'],
    training_array_task=os.environ['SLURM_ARRAY_TASK_ID'], inputs_sha256=sha(HERE/'inputs.json'),
    CPU_gate_sha256=sha(ROOT/lock['CPU_gate']))
path.write_text(json.dumps(fact, indent=2)+'\n')
command = [sys.executable, str(ROOT/'experiments/original-paper-scenarios/original-source-rate-code-008/train.py'),
    '--scope', 'engineering', '--stage', '1', '--rate', str(rate)]
os.execv(sys.executable, command)
