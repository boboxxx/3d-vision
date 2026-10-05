"""Only export all completed stage1 states after actual whole native audit."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def check_locks():
    lock = read(HERE/'inputs.json')
    assert all(sha(ROOT/p) == v for p, v in lock['files'].items())
    return lock


def main():
    check_locks()
    terminal_rows = {}
    for job in ['11426592', '11426847']:
        raw = subprocess.check_output(['sacct', '-j', job, '-n', '-P',
                                      '--format=JobIDRaw,State,ExitCode'], text=True)
        rows = [line.split('|') for line in raw.splitlines()]
        assert all([v, 'COMPLETED', '0:0'] in rows for v in [job, job+'.0']), raw
        terminal_rows[job] = raw
    directory = ROOT/'data/runs/original-public30-formal-seed17-003-stage1'
    report = read(directory/'report.json')
    native_path = ROOT/'data/provenance/original-public30-stage1-native-audit-007.json'
    terminal_path = ROOT/'data/provenance/original-public30-stage1-GPU-terminal-007.json'
    native = read(native_path)
    assert report['state'] == 'finished_complete_public_semantic_stage_pending_actual_terminal_and_audit'
    assert report['scope'] == 'formal' and report['stage'] == 1 and report['seed'] == 17
    assert report['job_id'] == '11426592' and report['update_count'] == 44544
    assert len(report['completed_epochs']) == 12
    assert native['state'] == 'passed_complete_original_public_semantic_stage'
    assert native['scope'] == 'formal' and native['stage'] == 1 and native['updates'] == 44544
    assert native['mode'] == 'native' and native['actual_terminal_closed'] and len(native['epochs']) == 12
    assert native['report_sha256'] == sha(directory/'report.json')
    assert native['records_sha256'] == report['records_sha256'] == sha(directory/'updates.jsonl')
    assert native['input_sha256'] == report['input_sha256'] == sha(ROOT/'experiments/original-paper-scenarios/original-public-train-code-003/inputs.json')
    assert native['verifier_sha256'] == sha(ROOT/'experiments/original-paper-scenarios/original-public-train-code-003/audit.py')
    assert native['terminal_sha256'] == sha(terminal_path)
    assert report['initial_sha256'] == sha(directory/'initial.pt')
    files = {'report.json': directory/'report.json', 'updates.jsonl': directory/'updates.jsonl',
             'initial.pt': directory/'initial.pt', 'native.json': native_path, 'terminal.json': terminal_path}
    for epoch, entry in enumerate(report['completed_epochs'], 1):
        assert entry['epoch'] == epoch and entry['samples'] == 3712 and entry['updates'] == epoch*3712
        path = directory/f'checkpoint-epoch{epoch}.pt'
        assert entry['path'] == str(path.relative_to(ROOT)) and entry['sha256'] == sha(path)
        assert native['epochs'][epoch-1]['checkpoint_sha256'] == entry['sha256']
        files[path.name] = path
    execution = dict(actual_export_job_id=os.environ['SLURM_JOB_ID'],
                     inputs_sha256=sha(HERE/'inputs.json'), source_actual_terminal_rows=terminal_rows,
                     report_sha256=sha(directory/'report.json'), native_sha256=sha(native_path),
                     checked_unix=time.time())
    execution_bytes = (json.dumps(execution, indent=2)+'\n').encode()
    entries = {name: dict(bytes=p.stat().st_size, sha256=sha(p)) for name, p in files.items()}
    entries['execution.json'] = dict(bytes=len(execution_bytes), sha256=hashlib.sha256(execution_bytes).hexdigest())
    assert all(0 < v['bytes'] <= 512*1024**2 for v in entries.values())
    assert sum(v['bytes'] for v in entries.values()) <= 5*1024**3//2
    index = (json.dumps(dict(entries=entries, inputs_sha256=sha(HERE/'inputs.json')), indent=2)+'\n').encode()

    def memory_member(archive, name, blob):
        info = tarfile.TarInfo(name); info.size = len(blob)
        archive.addfile(info, io.BytesIO(blob))

    with tarfile.open(fileobj=sys.stdout.buffer, mode='w|') as archive:
        memory_member(archive, '__index__.json', index)
        for name, path in files.items():
            assert sha(path) == entries[name]['sha256']
            info = tarfile.TarInfo(name); info.size = entries[name]['bytes']
            with path.open('rb') as source:
                archive.addfile(info, source)
        memory_member(archive, 'execution.json', execution_bytes)
    check_locks()


if __name__ == '__main__':
    main()
