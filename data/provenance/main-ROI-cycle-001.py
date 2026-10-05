"""Unique CPU-only ROI extraction/audit cycle and terminal artifact closure."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/sheng/paper6')
CODE = 'experiments/original-paper-scenarios/main-ROI-code'
sys.path.insert(0, str(ROOT / CODE))
import contract as c


def checks(scope):
    source = c.sources()
    for host in ('local', 'sheng'):
        proof = c.read(ROOT / 'data/engineering' / ('public-val-ROI-' + host + '-CPU-001.json'))
        assert proof['state'] == 'passed_four_sensor_ROI_CPU_families' and proof['families'] == 4
        assert proof['sources'] == source
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    if scope == 'main':
        proof = c.read(ROOT / 'data/provenance/public-val-ROI-engineering-001-local-verification.json')
        assert proof['state'] == 'passed_all4_transferred_ROI_view_records' and proof['sources'] == source
    return source


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--scope', choices=('engineering', 'main'), required=True)
    parser.add_argument('--mode', choices=('launch', 'run', 'close'), required=True); args = parser.parse_args()
    prefix = 'public-val-ROI-' + args.scope + '-001'; script = Path(__file__).resolve()
    launch = ROOT / 'data/runs' / (prefix + '-launch.json'); cycle = ROOT / 'data/runs' / (prefix + '-cycle.json')
    closure = ROOT / 'data/provenance' / (prefix + '-closure.json')
    manifest = ROOT / 'data/runs' / (prefix + '.json'); audit = ROOT / 'data/provenance' / (prefix + '-audit.json')
    if args.mode == 'launch':
        source = checks(args.scope); log = ROOT / 'logs' / (prefix + '-controller.log')
        assert not any(p.exists() for p in (launch, cycle, closure, manifest, audit, log, c.NATIVE / prefix))
        command = [sys.executable, str(script), '--mode', 'run', '--scope', args.scope]
        with log.open('x') as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        result = dict(state='launched_once', scope=args.scope, prefix=prefix, pid=child.pid, sources=source,
                      controller_sha256=c.sha(script), command=command, log=str(log), launched_unix=time.time(), GPU_used=False)
        with launch.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        print(json.dumps(dict(state=result['state'], pid=child.pid, scope=args.scope))); return
    if args.mode == 'run':
        source = checks(args.scope); assert not cycle.exists()
        run = dict(state='running', pid=os.getpid(), scope=args.scope, sources=source, controller_sha256=c.sha(script),
                   started_unix=time.time(), completed_commands=[], GPU_used=False)
        c.save(cycle, run)
        try:
            for index, name in enumerate(('extract', 'audit')):
                command = [sys.executable, CODE + '/' + name + '.py', '--scope', args.scope]
                log = ROOT / 'logs' / (prefix + '-command-' + str(index) + '.log'); started = time.time()
                with log.open('x') as stream:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT); status = child.wait()
                run['completed_commands'].append(dict(command=command, pid=child.pid, returncode=status,
                                                      started_unix=started, ended_unix=time.time(),
                                                      log=str(log), log_sha256=c.sha(log)))
                c.save(cycle, run); assert status == 0, command
            assert c.read(audit)['state'] == 'passed_all_native_PNG_pixels_and_independent_ROI_union_masks'
            assert c.sources() == source
            run.update(state='finished_CPU_ROI_and_audit_pending_terminal_closure', ended_unix=time.time())
        except BaseException as error:
            run.update(state='failed', error=repr(error), ended_unix=time.time()); raise
        finally:
            c.save(cycle, run)
        return
    assert not closure.exists(); start, run = c.read(launch), c.read(cycle)
    process = subprocess.run(['ps', '-p', str(start['pid']), '-o', 'pid,stat,args'], capture_output=True, text=True)
    assert process.returncode == 1 and start['pid'] == run['pid']
    assert run['state'] == 'finished_CPU_ROI_and_audit_pending_terminal_closure'
    assert start['sources'] == run['sources'] == c.sources()
    assert start['controller_sha256'] == run['controller_sha256'] == c.sha(script)
    assert len(run['completed_commands']) == 2 and all(row['returncode'] == 0 for row in run['completed_commands'])
    record = c.read(manifest); report = c.read(audit)
    assert record['state'] == 'finished' and report['state'] == 'passed_all_native_PNG_pixels_and_independent_ROI_union_masks'
    assert record['sources'] == report['sources'] == c.sources()
    assert record['initial_states'] == record['final_states']
    assert report['manifest_sha256'] == c.sha(manifest) and report['records_sha256'] == c.sha(record['records_path'])
    paths = {launch, cycle, manifest, audit, Path(start['log']), Path(record['records_path'])}
    for command in run['completed_commands']:
        assert c.sha(command['log']) == command['log_sha256']; paths.add(Path(command['log']))
    def mapped(path):
        if path.is_relative_to(ROOT):
            return str(path.relative_to(ROOT))
        assert path.is_relative_to(c.NATIVE / prefix)
        return 'data/engineering/' + prefix + '-native-transfer/' + str(path).lstrip('/')
    assert len(paths) == 8
    result = dict(state='closed_CPU_ROI_actual_terminal_all_native_artifacts', checked_unix=time.time(),
                  actual_terminal=True, scope=args.scope, pid=start['pid'], sources=c.sources(),
                  frames=report['frames'], views=report['views'], GPU_used=False,
                  artifacts_sha256={mapped(path): c.sha(path) for path in sorted(paths)},
                  native_to_local_artifacts={mapped(path): str(path) for path in sorted(paths)})
    with closure.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(state=result['state'], frames=result['frames'], views=result['views'], artifacts=8)))


if __name__ == '__main__':
    main()
