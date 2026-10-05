"""Create exclusive whole stage1 admission from actual full two-host proofs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
JOBS = ['11426592', '11426847', '11429371']
RUN = 'data/runs/original-public30-formal-seed17-003-stage1'
NATIVE = 'data/provenance/original-public30-stage1-native-audit-007.json'
LOCAL = 'data/provenance/original-public30-stage1-local-audit-008.json'
TERMINAL = 'data/provenance/original-public30-stage1-GPU-terminal-007.json'
GATE = 'data/provenance/original-public30-stage1-native-local-closure-009.json'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def locks():
    lock = read(HERE/'inputs.json')
    assert all(sha(ROOT/p) == v for p, v in lock['files'].items())


def report_check(report):
    assert report['state'] == 'finished_complete_public_semantic_stage_pending_actual_terminal_and_audit'
    assert report['scope'] == 'formal' and report['stage'] == 1 and report['seed'] == 17
    assert report['job_id'] == JOBS[0] and report['update_count'] == 44544
    assert len(report['completed_epochs']) == 12
    assert report['input_sha256'] == sha(ROOT/'experiments/original-paper-scenarios/original-public-train-code-003/inputs.json')
    for epoch, entry in enumerate(report['completed_epochs'], 1):
        assert entry['epoch'] == epoch and entry['samples'] == 3712 and entry['updates'] == epoch*3712
        assert entry['path'] == RUN+f'/checkpoint-epoch{epoch}.pt'


def proof_check(proof, role, report_sha, records_sha, terminal_sha):
    assert proof['state'] == 'passed_complete_original_public_semantic_stage'
    assert proof['scope'] == 'formal' and proof['stage'] == 1 and proof['mode'] == role
    assert proof['actual_terminal_closed'] and proof['updates'] == 44544 and len(proof['epochs']) == 12
    assert proof['report_sha256'] == report_sha and proof['records_sha256'] == records_sha
    assert proof['terminal_sha256'] == terminal_sha
    assert proof['input_sha256'] == sha(ROOT/'experiments/original-paper-scenarios/original-public-train-code-003/inputs.json')
    assert proof['verifier_sha256'] == sha(ROOT/'experiments/original-paper-scenarios/original-public-train-code-003/audit.py')


def collect():
    locks()
    accounting = {}
    for job in JOBS:
        raw = subprocess.check_output(['sacct', '-j', job, '-n', '-P',
                                      '--format=JobIDRaw,State,ExitCode'], text=True)
        rows = [line.split('|') for line in raw.splitlines()]
        assert all([v, 'COMPLETED', '0:0'] in rows for v in [job, job+'.0']), raw
        accounting[job] = raw
    directory = ROOT/RUN; report = read(directory/'report.json'); report_check(report)
    native = read(ROOT/NATIVE)
    proof_check(native, 'native', sha(directory/'report.json'), sha(directory/'updates.jsonl'), sha(ROOT/TERMINAL))
    assert report['records_sha256'] == native['records_sha256']
    files = {'report.json': directory/'report.json', 'updates.jsonl': directory/'updates.jsonl',
             'initial.pt': directory/'initial.pt', 'native.json': ROOT/NATIVE, 'terminal.json': ROOT/TERMINAL}
    assert sha(directory/'initial.pt') == report['initial_sha256']
    for epoch, entry in enumerate(report['completed_epochs'], 1):
        path = ROOT/entry['path']; assert sha(path) == entry['sha256']
        assert native['epochs'][epoch-1]['checkpoint_sha256'] == entry['sha256']
        files[path.name] = path
    result = dict(inputs_sha256=sha(HERE/'inputs.json'), accounting=accounting,
                  files={name: dict(bytes=p.stat().st_size, sha256=sha(p)) for name, p in files.items()},
                  native_utf8=(ROOT/NATIVE).read_text(), terminal_utf8=(ROOT/TERMINAL).read_text(),
                  checked_unix=time.time())
    locks(); print(json.dumps(result))


def close():
    locks(); out = ROOT/GATE; assert not out.exists()
    launcher = read(ROOT/'data/provenance/original-public30-stage1-local-launch-008.json')
    assert launcher['state'] == 'whole_local_exit0_pending_actual_export_terminal_and_stage_closure'
    assert launcher['local_auditor_exit_code'] == launcher['ssh_export_exit_code'] == 0
    assert launcher['actual_export_job_id'] == JOBS[2]
    assert launcher['inputs_sha256'] == sha(ROOT/'experiments/original-paper-scenarios/original-stage1-local-code-008/inputs.json')
    directory = ROOT/'data/engineering/original-public30-stage1-transfer-008'
    index = read(directory/'transfer-index.json')
    assert launcher['transfer_index_sha256'] == sha(directory/'transfer-index.json')
    expected_names = {'report.json', 'updates.jsonl', 'initial.pt', 'native.json', 'terminal.json', 'execution.json'}
    expected_names |= {f'checkpoint-epoch{x}.pt' for x in range(1, 13)}
    assert set(index['entries']) == expected_names
    for name, entry in index['entries'].items():
        assert (directory/name).stat().st_size == entry['bytes'] and sha(directory/name) == entry['sha256']
    command = ('cd /mnt/nfs2/engdes/wc296/paper6 && PYTHONDONTWRITEBYTECODE=1 '
               'envs/torch-cu128-001/bin/python experiments/original-paper-scenarios/original-stage2-dispatch-code-009/close_stage1.py --collect')
    c = json.loads(subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
        '-o', 'ControlPath=/private/tmp/hitac-artemis.sock', 'artemis-ood', command], text=True))
    assert c['inputs_sha256'] == sha(HERE/'inputs.json')
    assert set(c['files']) == expected_names-{'execution.json'}
    assert all(index['entries'][name] == entry for name, entry in c['files'].items())
    assert c['native_utf8'].encode() == (directory/'native.json').read_bytes()
    assert c['terminal_utf8'].encode() == (directory/'terminal.json').read_bytes()
    report = read(directory/'report.json'); report_check(report)
    native = read(directory/'native.json'); local = read(ROOT/LOCAL)
    for proof, role in [(native, 'native'), (local, 'local')]:
        proof_check(proof, role, sha(directory/'report.json'), sha(directory/'updates.jsonl'), sha(directory/'terminal.json'))
    assert local['epochs'] == native['epochs']
    assert local['native_audit_sha256'] == sha(directory/'native.json')
    assert launcher['local_proof_sha256'] == sha(ROOT/LOCAL)
    for rel, name in [(NATIVE, 'native.json'), (TERMINAL, 'terminal.json')]:
        path = ROOT/rel; blob = (directory/name).read_bytes()
        if path.exists(): assert path.read_bytes() == blob
        else:
            with path.open('xb') as f: f.write(blob)
    gate = dict(state='closed_original_public30_stage1_all44544_all12_native_local_actual_normal_terminals',
                stage=1, seed=17, updates=44544, epoch_checkpoints=12, jobs=JOBS,
                input_sha256=local['input_sha256'], closure_inputs_sha256=sha(HERE/'inputs.json'),
                accounting=c['accounting'], report_sha256=sha(directory/'report.json'),
                records_sha256=sha(directory/'updates.jsonl'), native_backup_files=c['files'],
                proof_files={role: dict(path=rel, sha256=sha(ROOT/rel)) for role, rel in [('native', NATIVE), ('local', LOCAL)]},
                local_auditor_actual_exit_code=0, export_actual_exit_code=0,
                local_launch_sha256=sha(ROOT/'data/provenance/original-public30-stage1-local-launch-008.json'),
                limitation=local['limitation'], AP_or_radio_or_baseline_completion_claimed=False,
                checked_unix=time.time())
    locks()
    with out.open('x') as f: json.dump(gate, f, indent=2); f.write('\n')
    print(json.dumps(dict(state=gate['state'], gate_sha256=sha(out))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--collect', action='store_true'); group.add_argument('--close', action='store_true')
    a = p.parse_args(); collect() if a.collect else close()
