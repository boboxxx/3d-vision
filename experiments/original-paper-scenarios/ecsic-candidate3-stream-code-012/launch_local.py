"""Run a bounded candidate3 formal003 local pipe and preserve both actual exits."""
import hashlib
import json
import os
import shlex
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


lock = json.loads((HERE/'inputs.json').read_text())
for rel, expected in lock['files'].items():
    assert sha(ROOT/rel) == expected, rel
out = ROOT/lock['local_output']
assert not out.exists() and not out.with_suffix('.records.jsonl').exists()
statepath = ROOT/'data/provenance/ecsic-candidate3-local-stream-launch-012.json'
exportlog = ROOT/'data/provenance/ecsic-candidate3-local-export-012.log'
auditlog = ROOT/'data/provenance/ecsic-candidate3-local-audit-012.log'
remote = ('cd /mnt/nfs2/engdes/wc296/paper6 && '
          'export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" && '
          'srun --account=engdes --partition=short --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:20:00 '
          '--dependency=afterok:11426848 --job-name=paper6-ecsic-export012 '
          'envs/torch-cu128-001/bin/python experiments/original-paper-scenarios/ecsic-candidate3-stream-code-012/export.py')
command = ['ssh','artemis-ood',remote]
local = [sys.executable,str(ROOT/'data/provenance/audit-ecsic-KITTI-formal-003.py'),
         '--mode','stream','--candidate','3','--output',str(out)]
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='')
with statepath.open('x') as f:
    json.dump({'state':'starting_foreground_dependency_wait_not_passed','started_unix':time.time(),
               'inputs_sha256':sha(HERE/'inputs.json'),'remote_command':shlex.join(command),
               'local_command':shlex.join(local)},f,indent=2)
with exportlog.open('x') as elog, auditlog.open('x') as alog:
    sender = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=elog,env=env)
    try:
        reader = subprocess.Popen(local,stdin=sender.stdout,stdout=alog,stderr=alog,env=env,cwd=ROOT)
    finally:
        sender.stdout.close()
    code_reader = reader.wait()
    code_sender = sender.wait()
r = json.loads(statepath.read_text())
r.update(state='local_pipe_exit0_pending_actual_export_terminal_and_two_host_closure' if code_reader == code_sender == 0 else 'failed_retained',
         finished_unix=time.time(),local_reader_exit_code=code_reader,ssh_export_exit_code=code_sender,
         pipefail_exit_code=code_reader or code_sender)
statepath.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
sys.exit(0 if code_reader == code_sender == 0 else 1)
