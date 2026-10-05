"""Once-only frozen SRCNN receive/audit and complete actual-terminal closure."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / 'experiments/original-paper-scenarios/srcnn-native-cache-code'
sys.path.insert(0, str(CODE))
import common as c

PLAN = 'experiments/original-paper-scenarios/srcnn-native-cache-terminal-orchestration-001.md'


class Deferred(Exception):
    pass


def orchestration_sources():
    return {name: c.sha(ROOT / name) for name in (
        PLAN, 'data/provenance/srcnn-native-cache-cycle-001.py',
        'data/provenance/verify-srcnn-native-cache-local-001.py')}


def gone(pid):
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'pid,stat,args'], capture_output=True, text=True)
    if result.returncode == 0:
        return False
    assert result.returncode == 1, result.stderr
    return True


def formal_priority():
    main = ROOT / 'data/provenance/source-cache-inference-main-001-closure.json'
    if not main.exists():
        cycle = c.read(ROOT / 'data/runs/source-cache-inference-main-001-cycle.json')
        if cycle['state'] == 'failed':
            raise RuntimeError('main inference failed; retain its artifacts')
        raise Deferred('Main inference actual-terminal closure is pending')
    proof_path = ROOT / 'data/provenance/source-cache-inference-main-001-local-verification.json'
    if not proof_path.exists():
        raise Deferred('Complete local main prediction proof is pending')
    closed, proof = c.read(main), c.read(proof_path)
    assert closed['actual_terminal'] and closed['frames'] == 45228
    assert proof['closure_sha256'] == c.sha(main)
    assert proof['state'] == 'passed_all45228_transferred_native_inference_records_predictions_and_metrics'
    prefix = 'srcnn-kitti-formal-seed17-002'
    launch = ROOT / 'data/runs' / (prefix + '-launch.json')
    cycle = ROOT / 'data/runs' / (prefix + '-cycle.json')
    if not launch.exists() or not cycle.exists():
        raise Deferred('Formal SRCNN training has priority and has not yet completed')
    started, run = c.read(launch), c.read(cycle)
    if run['state'] == 'failed':
        raise RuntimeError('formal training failed; retain its artifacts')
    if not gone(started['pid']):
        raise Deferred('Formal SRCNN training/audit process is still live')
    assert run['state'] == 'finished_six_commands_pending_terminal_closure'
    assert run['pid'] == started['pid'] and len(run['completed_commands']) == 6
    assert all(row['returncode'] == 0 and gone(row['pid']) for row in run['completed_commands'])
    assert run['sources'] == started['sources'] == c.train.sources()
    assert run['controller_sha256'] == started['controller_sha256'] == c.sha(ROOT / 'data/provenance/srcnn-kitti-cycle-002.py')


def static_gate(scope):
    source = c.CPU_gate(); _, dependencies = c.gate(scope)
    parent_path = ROOT / 'data/runs' / (c.prefix(scope) + '-encode.json')
    parent = c.read(parent_path)
    assert parent['state'] == 'finished_all_three_source_wire_conditions'
    assert parent['sources'] == source and parent['dependencies'] == dependencies
    assert parent['scope'] == scope and parent['frame_ids'] == c.ids(scope)
    assert parent['pairs'] == 3 * len(parent['frame_ids'])
    if scope == 'engineering':
        audit_path = ROOT / 'data/provenance' / (c.prefix(scope) + '-wire-audit.json')
        proof = c.read(ROOT / 'data/provenance' / (c.prefix(scope) + '-wire-local-verification.json'))
        audit = c.read(audit_path)
        assert audit['actual_encoder_terminal'] and audit['encoder_manifest_sha256'] == c.sha(parent_path)
        assert proof['state'] == 'passed_all6_actual_transferred_SRCNN_engineering_wire_files_and8_artifacts'
        assert proof['native_wire_audit_sha256'] == c.sha(audit_path)
        assert proof['sources'] == audit['sources'] == source and proof['dependencies'] == audit['dependencies'] == dependencies
        for rate in (10, 30, 50):
            for row in parent['conditions'][str(rate)]['rows']:
                assert c.sha(row['wire_path']) == row['wire_sha256'] == audit['wire_files_sha256'][row['wire_path']]
    else:
        path = ROOT / 'data/provenance/srcnn-source-cache-engineering-001-closure.json'
        proof = c.read(ROOT / 'data/provenance/srcnn-source-cache-engineering-001-local-verification.json')
        assert c.read(path)['actual_terminal'] and proof['closure_sha256'] == c.sha(path)
        assert proof['state'] == 'passed_all6_transferred_native_SRCNN_received_pairs_and28_artifacts'
        assert proof['sources'] == source
    return source, dependencies, parent


def eligible(scope):
    source, dependencies, parent = static_gate(scope)
    formal_priority()
    free = c.cuda_gate()
    return source, dependencies, parent, free


def commands(scope):
    return [[sys.executable, str(CODE / (mode + '.py')), '--scope', scope, '--rate', str(rate)]
            for rate in (10, 30, 50) for mode in ('receive', 'audit')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('eligible', 'launch', 'run', 'close'), required=True)
    parser.add_argument('--scope', choices=('engineering', 'main'), required=True)
    args = parser.parse_args(); assert ROOT == Path('/home/sheng/paper6')
    prefix = c.prefix(args.scope); script = Path(__file__).resolve()
    launch = ROOT / 'data/runs' / (prefix + '-launch.json')
    cycle = ROOT / 'data/runs' / (prefix + '-cycle.json')
    closure = ROOT / 'data/provenance' / (prefix + '-closure.json')
    log = ROOT / 'logs' / (prefix + '-controller.log')
    identity = orchestration_sources()
    if args.mode == 'eligible':
        try:
            _, _, parent, free = eligible(args.scope)
            result = dict(state='eligible_without_dispatch', scope=args.scope, wire_pairs=parent['pairs'], free_bytes=free)
        except Deferred as error:
            result = dict(state='deferred_before_GPU_dispatch', scope=args.scope, reason=str(error))
        result.update(checked_unix=time.time(), CUDA_initialized=c.torch.cuda.is_initialized(), orchestration_sources=identity)
        print(json.dumps(result)); return
    if args.mode == 'launch':
        source, dependencies, parent, free = eligible(args.scope)
        paths = [launch, cycle, closure, log]
        paths += [ROOT / 'logs' / (prefix + '-command-' + str(i) + '.log') for i in range(6)]
        for rate in (10, 30, 50):
            paths += [ROOT / 'data/runs' / (prefix + '-cr' + str(rate) + '-receive.json'),
                      ROOT / 'data/provenance' / (prefix + '-cr' + str(rate) + '-audit.json'),
                      c.NATIVE / prefix / ('cr' + str(rate)) / 'received']
        assert not any(path.exists() for path in paths), 'Existing artifacts prohibit run-ID reuse'
        command = [sys.executable, str(script), '--mode', 'run', '--scope', args.scope]
        with log.open('x') as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        record = dict(state='launched_once', pid=child.pid, scope=args.scope, launched_unix=time.time(),
                      sources=source, dependencies=dependencies, orchestration_sources=identity,
                      encoding_manifest_sha256=c.sha(ROOT / 'data/runs' / (prefix + '-encode.json')),
                      physical_free_before_bytes=free, command=command, log=str(log))
        with launch.open('x') as stream: json.dump(record, stream, indent=2, allow_nan=False)
        print(json.dumps(dict(state=record['state'], pid=child.pid))); return
    if args.mode == 'run':
        assert not cycle.exists()
        record = dict(state='running', pid=os.getpid(), scope=args.scope, started_unix=time.time(),
                      orchestration_sources=identity, completed_commands=[], rates={})
        c.save(cycle, record)
        try:
            source, dependencies, parent, _ = eligible(args.scope)
            record.update(sources=source, dependencies=dependencies,
                          encoding_manifest_sha256=c.sha(ROOT / 'data/runs' / (prefix + '-encode.json')))
            c.save(cycle, record)
            for index, command in enumerate(commands(args.scope)):
                assert c.sources() == source and orchestration_sources() == identity
                command_log = ROOT / 'logs' / (prefix + '-command-' + str(index) + '.log')
                started = time.time(); record['current_command'] = index; c.save(cycle, record)
                with command_log.open('x') as stream:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                    status = child.wait()
                record['completed_commands'].append(dict(command=command, pid=child.pid, returncode=status,
                    started_unix=started, ended_unix=time.time(), log=str(command_log), log_sha256=c.sha(command_log)))
                c.save(cycle, record); assert status == 0, command
                if index % 2:
                    rate = (10, 30, 50)[index // 2]
                    receipt = ROOT / 'data/runs' / (prefix + '-cr' + str(rate) + '-receive.json')
                    audit = ROOT / 'data/provenance' / (prefix + '-cr' + str(rate) + '-audit.json')
                    assert c.read(audit)['state'] == 'passed_every_native_SRCNN_source_wire_float_pixel_and_readonly_state'
                    record['rates'][str(rate)] = dict(manifest=str(receipt), manifest_sha256=c.sha(receipt),
                                                      audit=str(audit), audit_sha256=c.sha(audit))
                    c.save(cycle, record)
            record.update(state='finished_six_commands_pending_actual_terminal_closure', ended_unix=time.time())
        except BaseException as error:
            record.update(state='failed', error=repr(error), ended_unix=time.time()); raise
        finally: c.save(cycle, record)
        return
    assert not closure.exists()
    start, run = c.read(launch), c.read(cycle)
    assert start['pid'] == run['pid'] and gone(start['pid'])
    assert run['state'] == 'finished_six_commands_pending_actual_terminal_closure'
    source, dependencies, parent = static_gate(args.scope)
    assert start['sources'] == run['sources'] == source and start['dependencies'] == run['dependencies'] == dependencies
    assert start['orchestration_sources'] == run['orchestration_sources'] == identity
    parent_path = ROOT / 'data/runs' / (prefix + '-encode.json')
    assert start['encoding_manifest_sha256'] == run['encoding_manifest_sha256'] == c.sha(parent_path)
    assert len(run['completed_commands']) == 6 and set(run['rates']) == {'10', '30', '50'}
    paths = {parent_path, launch, cycle, Path(start['log'])}; pixels = 0
    for expected, command in zip(commands(args.scope), run['completed_commands']):
        assert command['command'] == expected and command['returncode'] == 0 and gone(command['pid'])
        assert c.sha(command['log']) == command['log_sha256']; paths.add(Path(command['log']))
    for rate in (10, 30, 50):
        item = run['rates'][str(rate)]; manifest, audit_path = Path(item['manifest']), Path(item['audit'])
        receipt, audit = c.read(manifest), c.read(audit_path)
        assert c.sha(manifest) == item['manifest_sha256'] == audit['manifest_sha256']
        assert c.sha(audit_path) == item['audit_sha256']
        assert receipt['state'] == 'finished_all_native_received_pairs'
        assert audit['state'] == 'passed_every_native_SRCNN_source_wire_float_pixel_and_readonly_state'
        assert receipt['sources'] == audit['sources'] == source and receipt['dependencies'] == audit['dependencies'] == dependencies
        assert receipt['encoding_manifest_sha256'] == audit['encoding_manifest_sha256'] == c.sha(parent_path)
        assert receipt['model_initial_state'] == receipt['model_final_state'] == audit['readonly_states']
        assert audit['maximum_RGB_error'] <= 3e-5 and gone(receipt['pid'])
        assert receipt['ordered_ids'] == parent['frame_ids'] and len(receipt['frames']) == len(audit['rows']) == len(parent['frame_ids'])
        paths.update((manifest, audit_path)); count = 0
        for encoded, received, checked in zip(parent['conditions'][str(rate)]['rows'], receipt['frames'], audit['rows']):
            assert encoded['frame_id'] == received['frame_id'] == checked['frame_id']
            assert encoded['wire_path'] == received['wire_path']
            assert c.sha(encoded['wire_path']) == encoded['wire_sha256'] == received['wire_sha256'] == checked['wire_sha256']
            assert c.sha(received['cache_path']) == received['cache_sha256'] == checked['cache_sha256']
            assert received['arrays'] == checked['arrays'] and checked['independent_max_abs_error'] <= 3e-5
            count += sum(math.prod(value['shape']) for value in received['arrays'].values())
            paths.update((Path(encoded['wire_path']), Path(received['cache_path'])))
        assert count == audit['RGB_values']; pixels += count
    def mapped(path):
        if path.is_relative_to(ROOT): return str(path.relative_to(ROOT))
        assert path.is_relative_to(c.NATIVE / prefix)
        return 'data/engineering/' + prefix + '-native-transfer/' + str(path).lstrip('/')
    assert len(paths) == 16 + 6 * len(parent['frame_ids'])
    result = dict(state='closed_all_native_SRCNN_received_pairs_actual_terminal', scope=args.scope,
        actual_terminal=True, pid=start['pid'], checked_unix=time.time(), pairs=parent['pairs'], RGB_values=pixels,
        sources=source, dependencies=dependencies, orchestration_sources=identity,
        artifacts_sha256={mapped(path): c.sha(path) for path in sorted(paths)},
        native_to_local_artifacts={mapped(path): str(path) for path in sorted(paths)})
    with closure.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    for suffix, members in (('transfer-files', [path for path in paths if path.is_relative_to(ROOT)]),
                            ('native-transfer-files', [path for path in paths if not path.is_relative_to(ROOT)])):
        values = [str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path).lstrip('/') for path in sorted(members)]
        (ROOT / 'data/provenance' / (prefix + '-' + suffix + '.txt')).write_text('\n'.join(values) + '\n')
    print(json.dumps(dict(state=result['state'], pairs=result['pairs'], artifacts=len(paths))))


if __name__ == '__main__': main()
