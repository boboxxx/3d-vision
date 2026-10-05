"""One unique source/cache cycle; launcher, controller and terminal closure."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/home/sheng/paper6')
CODE = 'experiments/original-paper-scenarios/source-cache-code'
sys.path.insert(0, str(ROOT / CODE))
import common as c


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def checks(scope):
    c.native_environment()
    local = c.read(ROOT / 'data/engineering/source-cache-local-CPU-001.json')
    native = c.read(ROOT / 'data/engineering/source-cache-sheng-CPU-001.json')
    assert local['state'] == native['state'] == 'passed'
    assert local['tests'] == native['tests'] == 4
    assert local['sources'] == native['sources'] == c.identities()
    assert local['predecessor'] == native['predecessor'] == c.predecessor()
    if scope == 'main':
        proof = c.read(ROOT / 'data/provenance/original-source-cache-engineering-001-local-verification.json')
        assert proof['state'] == 'passed_all_transferred_records_and_engineering_wires'
        assert proof['pairs'] == 12 and proof['sources'] == c.identities()
        assert shutil.disk_usage('/mnt/d/paper6').free >= 100 * 1024**3
    return local['sources']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('launch', 'run', 'close'), required=True)
    p.add_argument('--scope', choices=('engineering', 'main'), required=True)
    args = p.parse_args()
    prefix = 'original-source-cache-' + args.scope + '-001'
    directory = Path('/mnt/d/paper6/runs') / prefix
    run = ROOT / 'data/runs' / (prefix + '-cycle.json')
    launch = ROOT / 'data/runs' / (prefix + '-launch.json')
    audit = ROOT / 'data/provenance' / (prefix + '-audit.json')
    closure = ROOT / 'data/provenance' / (prefix + '-closure.json')
    script = Path(__file__).resolve()
    if args.mode == 'launch':
        sources = checks(args.scope)
        log = ROOT / 'logs' / (prefix + '-controller.log')
        assert not any(x.exists() for x in (directory, run, launch, audit, closure, log))
        command = [sys.executable, str(script), '--mode', 'run', '--scope', args.scope]
        with log.open('x') as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        value = dict(state='launched_once', pid=child.pid, prefix=prefix, scope=args.scope,
                     launched_unix=time.time(), sources=sources, controller_sha256=c.sha(script),
                     command=command, log=str(log), directory=str(directory))
        with launch.open('x') as stream: json.dump(value, stream, indent=2)
        print(json.dumps(value)); return
    if args.mode == 'run':
        sources = checks(args.scope); os.nice(10)
        assert not run.exists() and not directory.exists()
        value = dict(state='running', pid=os.getpid(), scope=args.scope, prefix=prefix,
                     sources=sources, controller_sha256=c.sha(script), started_unix=time.time(), completed_commands=[])
        save(run, value)
        commands = [[sys.executable, CODE + '/encode.py', '--directory', str(directory), '--scope', args.scope]]
        commands += [[sys.executable, CODE + '/receive.py', '--directory', str(directory), '--scope', args.scope,
                      '--condition', x[0]] for x in c.CONDITIONS]
        commands += [[sys.executable, CODE + '/audit.py', '--directory', str(directory), '--scope', args.scope,
                      '--output', str(audit)]]
        try:
            for index, command in enumerate(commands):
                assert c.identities() == sources
                log = ROOT / 'logs' / (prefix + '-command-' + str(index) + '.log')
                value['current_command'] = index; save(run, value)
                started = time.time()
                with log.open('x') as stream:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                    status = child.wait()
                record = dict(command=command, pid=child.pid, started_unix=started, ended_unix=time.time(),
                              returncode=status, log=str(log), log_sha256=c.sha(log))
                value['completed_commands'].append(record); save(run, value)
                assert status == 0, record
            result = c.read(audit)
            assert result['pairs'] == (12 if args.scope == 'engineering' else 22614)
            assert result['state'] == 'passed_all_six_complete_conditions_full_native_pixels_and_sources'
            assert c.identities() == sources and c.sha(script) == value['controller_sha256']
            value.update(state='finished_eight_commands_pending_terminal_closure', ended_unix=time.time(), audit_sha256=c.sha(audit))
        except BaseException as error:
            value.update(state='failed', ended_unix=time.time(), error=repr(error)); raise
        finally: save(run, value)
        return
    assert not closure.exists()
    start = c.read(launch); cycle = c.read(run); result = c.read(audit)
    ps = subprocess.run(['ps', '-p', str(start['pid']), '-o', 'pid,stat,args'], text=True, capture_output=True)
    assert ps.returncode == 1 and cycle['pid'] == start['pid']
    assert cycle['state'] == 'finished_eight_commands_pending_terminal_closure'
    assert len(cycle['completed_commands']) == 8 and all(x['returncode'] == 0 for x in cycle['completed_commands'])
    assert start['sources'] == cycle['sources'] == result['sources'] == c.identities()
    assert start['controller_sha256'] == cycle['controller_sha256'] == c.sha(script)
    assert cycle['audit_sha256'] == c.sha(audit)
    assert all(c.sha(path) == digest for path, digest in result['native_files_sha256'].items())
    transferred = [launch, run, audit, Path(start['log'])] + [Path(x['log']) for x in cycle['completed_commands']]
    transferred += [directory / 'encode.json'] + [directory / 'received' / x[0] / 'receiver.json' for x in c.CONDITIONS]
    if args.scope == 'engineering':
        transferred += [directory / 'source' / key / (frame + '.p6sb') for key, *_ in c.CONDITIONS for frame in c.frames('engineering')]
    value = dict(state='closed_actual_terminal_all_native_artifacts', actual_terminal=True, checked_unix=time.time(),
                 pid=start['pid'], scope=args.scope, pairs=result['pairs'], views=result['views'], sources=c.identities(),
                 native_files=result['native_files_sha256'], transferred_files={str(x): c.sha(x) for x in transferred})
    with closure.open('x') as stream: json.dump(value, stream, indent=2)
    print(json.dumps(dict(state=value['state'], pairs=value['pairs'], transfer_files=len(transferred))))


if __name__ == '__main__': main()
