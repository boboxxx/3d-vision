"""Foreground bounded complete local003 audit for one final candidate."""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from native import ROOT, HERE, LAMBDAS, sha, locked

parser = argparse.ArgumentParser()
parser.add_argument('--candidate', type=int, choices=[4, 5], required=True)
parser.add_argument('--native-array', required=True)
args = parser.parse_args()
assert args.native_array.isdigit()
locked()
candidate = args.candidate
out = ROOT/'data/provenance'/f'ecsic-KITTI-lambda{LAMBDAS[candidate]:.3g}-seed17-002-local-audit-003.json'
assert not out.exists()
path = ROOT/'data/provenance'/f'ecsic-candidate{candidate}-local-stream-launch-013.json'
elog = ROOT/'data/provenance'/f'ecsic-candidate{candidate}-local-export-013.log'
alog = ROOT/'data/provenance'/f'ecsic-candidate{candidate}-local-audit-013.log'
remote = ('cd /mnt/nfs2/engdes/wc296/paper6 && '
    'export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" && '
    'srun --account=engdes --partition=short --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:20:00 '
    f'--dependency=afterok:{args.native_array}_{candidate} --job-name=paper6-ecsic-c{candidate}-export013 '
    'envs/torch-cu128-001/bin/python experiments/original-paper-scenarios/ecsic-final-candidates-code-013/export.py '
    f'--candidate {candidate} --native-array {args.native_array}')
command = ['ssh', 'artemis-ood', remote]
local = [sys.executable, str(ROOT/'data/provenance/audit-ecsic-KITTI-formal-003.py'),
    '--mode', 'stream', '--candidate', str(candidate), '--output', str(out)]
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='')
fact = dict(state='starting_foreground_dependency_wait_not_passed', candidate=candidate,
    native_array_job=args.native_array, started_unix=time.time(), inputs_sha256=sha(HERE/'inputs.json'),
    remote_command=shlex.join(command), local_command=shlex.join(local))
with path.open('x') as f:
    json.dump(fact, f, indent=2)
with elog.open('x') as e, alog.open('x') as a:
    sender = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=e, env=env)
    try:
        reader = subprocess.Popen(local, stdin=sender.stdout, stdout=a, stderr=a, env=env, cwd=ROOT)
    finally:
        sender.stdout.close()
    reader_code, sender_code = reader.wait(), sender.wait()
fact.update(state='local_pipe_exit0_pending_actual_export_terminal_and_two_host_closure' if reader_code == sender_code == 0 else 'failed_retained',
    finished_unix=time.time(), local_reader_exit_code=reader_code, ssh_export_exit_code=sender_code,
    pipefail_exit_code=reader_code or sender_code)
path.write_text(json.dumps(fact, indent=2)+'\n')
print(json.dumps(fact))
sys.exit(fact['pipefail_exit_code'])
