"""Once-only CPU full-main quality, audit and terminal-data closure sequence."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/sheng/paper6')
SCRIPT = ROOT / 'data/provenance/native-source-quality-001.py'
spec = importlib.util.spec_from_file_location('main_quality_pipeline', SCRIPT)
q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
PREFIX = 'source-quality-main-001'


def checks():
    source, _, _, _, _, _ = q.gate('main')
    proof = q.r.read(ROOT / 'data/provenance/source-quality-engineering-001-local-verification.json')
    assert proof['state'] == 'passed_all24_transferred_native_quality_records' and proof['sources'] == source
    engineering = ROOT / 'data/provenance/source-quality-engineering-001-closure.json'
    assert proof['closure_sha256'] == q.r.sha(engineering) and q.r.read(engineering)['actual_terminal']
    return source


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', choices=('launch', 'run'), required=True)
    args = parser.parse_args(); source = checks(); script = Path(__file__).resolve()
    launch = ROOT / 'data/runs' / (PREFIX + '-launch.json'); cycle = ROOT / 'data/runs' / (PREFIX + '-cycle.json')
    if args.mode == 'launch':
        log = ROOT / 'logs' / (PREFIX + '-controller.log')
        manifest = ROOT / 'data/runs' / (PREFIX + '.json'); audit = ROOT / 'data/provenance' / (PREFIX + '-audit.json')
        closure = ROOT / 'data/provenance' / (PREFIX + '-closure.json')
        assert not any(path.exists() for path in (launch, cycle, log, manifest, audit, closure, q.r.NATIVE / PREFIX))
        command = [sys.executable, str(script), '--mode', 'run']
        with log.open('x') as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        record = dict(state='launched_once', pid=child.pid, sources=source, controller_sha256=q.r.sha(script),
                      launched_unix=time.time(), command=command, log=str(log), GPU_used=False)
        with launch.open('x') as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
        print(json.dumps(dict(state=record['state'], pid=child.pid, GPU_used=False))); return
    assert not cycle.exists()
    record = dict(state='running', pid=os.getpid(), started_unix=time.time(), sources=source,
                  controller_sha256=q.r.sha(script), completed_commands=[], GPU_used=False)
    q.r.save(cycle, record)
    commands = [[sys.executable, str(SCRIPT), '--scope', 'main', '--mode', mode] for mode in ('measure', 'audit')]
    commands.append([sys.executable, str(ROOT / 'data/provenance/close-source-quality-native-001.py'), '--scope', 'main'])
    try:
        for index, command in enumerate(commands):
            assert q.sources() == source
            log = ROOT / 'logs' / (PREFIX + '-command-' + str(index) + '.log'); started = time.time()
            with log.open('x') as stream:
                child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT); status = child.wait()
            record['completed_commands'].append(dict(command=command, pid=child.pid, returncode=status,
                                                      started_unix=started, ended_unix=time.time(),
                                                      log=str(log), log_sha256=q.r.sha(log)))
            q.r.save(cycle, record); assert status == 0, command
        closed = q.r.read(ROOT / 'data/provenance' / (PREFIX + '-closure.json'))
        assert closed['state'] == 'closed_native_quality_actual_terminal_all_records' and closed['actual_terminal']
        assert closed['views'] == 45228
        record.update(state='finished_all_three_commands_native_quality_closed', ended_unix=time.time(), views=45228)
    except BaseException as error:
        record.update(state='failed', error=repr(error), ended_unix=time.time()); raise
    finally:
        q.r.save(cycle, record)


if __name__ == '__main__':
    main()
