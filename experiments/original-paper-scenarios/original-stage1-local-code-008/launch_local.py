"""Exclusive complete state transfer followed by unchanged whole auditor003."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    lock = json.loads((HERE/'inputs.json').read_text())
    assert all(sha(ROOT/p) == v for p, v in lock['files'].items())
    directory = ROOT/'data/engineering/original-public30-stage1-transfer-008'
    out = ROOT/'data/provenance/original-public30-stage1-local-audit-008.json'
    statepath = ROOT/'data/provenance/original-public30-stage1-local-launch-008.json'
    exportlog = ROOT/'data/provenance/original-public30-stage1-local-export-008.log'
    auditlog = ROOT/'data/provenance/original-public30-stage1-local-audit-008.log'
    assert not any(p.exists() for p in [directory, out, statepath, exportlog, auditlog])
    directory.mkdir()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='')
    remote = ('cd /mnt/nfs2/engdes/wc296/paper6 && '
              'export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" && '
              'srun --account=engdes --partition=short --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:30:00 '
              '--dependency=afterok:11426847 --job-name=paper6-original-local008 '
              'envs/torch-cu128-001/bin/python experiments/original-paper-scenarios/original-stage1-local-code-008/export.py')
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
               '-o', 'ControlPath=/private/tmp/hitac-artemis.sock', 'artemis-ood', remote]
    result = dict(state='foreground_waiting_native_parent_not_passed', started_unix=time.time(),
                  inputs_sha256=sha(HERE/'inputs.json'), remote_command=command)
    with statepath.open('x') as f:
        json.dump(result, f, indent=2)
    sender = None
    try:
        with exportlog.open('x') as elog:
            sender = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=elog, env=env)
            with tarfile.open(fileobj=sender.stdout, mode='r|') as archive:
                member = archive.next()
                assert member is not None and member.name == '__index__.json' and member.isfile() and member.size <= 1024**2
                index_blob = archive.extractfile(member).read()
                index = json.loads(index_blob)
                assert index['inputs_sha256'] == sha(HERE/'inputs.json')
                entries = index['entries']
                names = {'report.json', 'updates.jsonl', 'initial.pt', 'native.json', 'terminal.json', 'execution.json'}
                names |= {f'checkpoint-epoch{x}.pt' for x in range(1, 13)}
                assert set(entries) == names
                assert all(0 < v['bytes'] <= 512*1024**2 and len(v['sha256']) == 64 for v in entries.values())
                total = sum(v['bytes'] for v in entries.values())
                assert total <= 5*1024**3//2 and shutil.disk_usage(ROOT).free >= total+512*1024**2
                (directory/'transfer-index.json').write_bytes(index_blob)
                seen = set()
                while (member := archive.next()) is not None:
                    assert member.isfile() and member.name in names and member.name not in seen
                    expected = entries[member.name]; assert member.size == expected['bytes']
                    digest = hashlib.sha256(); copied = 0
                    source = archive.extractfile(member)
                    with (directory/member.name).open('xb') as target:
                        while copied < member.size:
                            block = source.read(min(8*1024**2, member.size-copied)); assert block
                            target.write(block); digest.update(block); copied += len(block)
                    assert copied == expected['bytes'] and digest.hexdigest() == expected['sha256']
                    seen.add(member.name)
                assert seen == names
            sender.stdout.close()
            result['ssh_export_exit_code'] = sender.wait()
            assert result['ssh_export_exit_code'] == 0
        execution = json.loads((directory/'execution.json').read_text())
        result['actual_export_job_id'] = execution['actual_export_job_id']
        assert execution['inputs_sha256'] == sha(HERE/'inputs.json')
        assert all(sha(directory/name) == entry['sha256'] for name, entry in entries.items())
        audit = [sys.executable, str(ROOT/'experiments/original-paper-scenarios/original-public-train-code-003/audit.py'),
                 '--directory', str(directory), '--terminal', str(directory/'terminal.json'),
                 '--native-audit', str(directory/'native.json'), '--output', str(out)]
        with auditlog.open('x') as log:
            result['local_auditor_exit_code'] = subprocess.run(audit, env=env, cwd=ROOT, stdout=log, stderr=log).returncode
        assert result['local_auditor_exit_code'] == 0
        assert all(sha(directory/name) == entry['sha256'] for name, entry in entries.items())
        assert all(sha(ROOT/p) == v for p, v in lock['files'].items())
        result.update(state='whole_local_exit0_pending_actual_export_terminal_and_stage_closure',
                      transfer_index_sha256=sha(directory/'transfer-index.json'), local_proof_sha256=sha(out),
                      complete_records=44544, complete_epoch_states=12, transferred_bytes=total)
    except BaseException:
        result.update(state='failed_retained', traceback=traceback.format_exc())
        if sender is not None:
            if sender.stdout is not None:
                sender.stdout.close()
            result['ssh_export_exit_code'] = sender.wait()
        raise
    finally:
        result['finished_unix'] = time.time()
        statepath.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
