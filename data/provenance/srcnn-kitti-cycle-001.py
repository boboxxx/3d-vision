"""Once-only SRCNN three-rate training/audit cycle with actual terminal closure."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/sheng/paper6')
CODE = 'experiments/original-paper-scenarios/srcnn-KITTI-adaptation-code'
sys.path.insert(0, str(ROOT / CODE))
import common as c


def checks(scope):
    sources = c.sources()
    local = c.read(ROOT / 'data/engineering/srcnn-KITTI-adaptation-local-CPU-001.json')
    native = c.read(ROOT / 'data/engineering/srcnn-KITTI-adaptation-sheng-CPU-001.json')
    assert local['state'] == native['state'] == 'passed_four_SRCNN_training_CPU_families'
    assert local['families'] == native['families'] == 4
    assert local['sources'] == native['sources'] == sources
    assert local['author_initial_float32_states'] == native['author_initial_float32_states']
    f9 = c.read(ROOT / 'data/provenance/stereo-epipolar-native-seed17-001-closure.json')
    assert f9['actual_terminal']
    if scope == 'formal':
        closure_path = ROOT / 'data/provenance/srcnn-kitti-engineering-seed17-001-closure.json'
        engineering = c.read(closure_path)
        proof = c.read(ROOT / 'data/provenance/srcnn-kitti-engineering-seed17-001-local-verification.json')
        assert engineering['actual_terminal'] and engineering['sources'] == sources
        assert proof['state'] == 'passed_all18_transferred_SRCNN_training_records_and_terminal_artifacts'
        assert proof['closure_sha256'] == c.sha(closure_path)
        main_path = ROOT / 'data/provenance/source-cache-inference-main-001-closure.json'
        main = c.read(main_path)
        main_proof = c.read(ROOT / 'data/provenance/source-cache-inference-main-001-local-verification.json')
        assert main['actual_terminal'] and main['frames'] == 45228
        assert main_proof['closure_sha256'] == c.sha(main_path)
        assert main_proof['state'] == 'passed_all45228_transferred_native_inference_records_predictions_and_metrics'
    else:
        # Engineering finishes before main inference is dispatched; don't compete.
        path = ROOT / 'data/runs/source-cache-inference-main-001-launch.json'
        if path.exists():
            result = subprocess.run(['ps', '-p', str(c.read(path)['pid'])], capture_output=True)
            assert result.returncode == 1, 'main detector cycle still live'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip()) * 2**20
    assert free >= 12 * 2**30
    return sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('launch', 'run', 'close'), required=True)
    parser.add_argument('--scope', choices=('engineering', 'formal'), required=True)
    args = parser.parse_args()
    prefix = f'srcnn-kitti-{args.scope}-seed17-001'
    launch = ROOT / 'data/runs' / (prefix + '-launch.json')
    cycle = ROOT / 'data/runs' / (prefix + '-cycle.json')
    closure = ROOT / 'data/provenance' / (prefix + '-closure.json')
    script = Path(__file__).resolve()
    if args.mode == 'launch':
        sources = checks(args.scope)
        log = ROOT / 'logs' / (prefix + '-controller.log')
        assert not any(path.exists() for path in (launch, cycle, closure, log))
        for rate in (10, 30, 50):
            name = prefix + '-cr' + str(rate)
            assert not (ROOT / 'data/runs' / (name + '.json')).exists() and not (c.NATIVE / name).exists()
        command = [sys.executable, str(script), '--mode', 'run', '--scope', args.scope]
        with log.open('x') as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        result = dict(state='launched_once', prefix=prefix, scope=args.scope, pid=child.pid,
                      launched_unix=time.time(), sources=sources, controller_sha256=c.sha(script), command=command, log=str(log))
        with launch.open('x') as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps(result)); return
    if args.mode == 'run':
        sources = checks(args.scope); assert not cycle.exists()
        value = dict(state='running', prefix=prefix, scope=args.scope, pid=os.getpid(), started_unix=time.time(),
                     sources=sources, controller_sha256=c.sha(script), completed_commands=[], rates={})
        c.save(cycle, value)
        try:
            for rate in (10, 30, 50):
                name = prefix + '-cr' + str(rate)
                manifest = ROOT / 'data/runs' / (name + '.json')
                audit = ROOT / 'data/provenance' / (name + '-audit.json')
                commands = [[sys.executable, CODE + '/train.py', '--scope', args.scope, '--rate', str(rate)],
                            [sys.executable, CODE + '/audit.py', '--manifest', str(manifest), '--output', str(audit)]]
                for command in commands:
                    assert c.sources() == sources
                    index = len(value['completed_commands']); value.update(current_rate=rate, current_command=index)
                    c.save(cycle, value)
                    log = ROOT / 'logs' / (prefix + '-command-' + str(index) + '.log')
                    started = time.time()
                    with log.open('x') as stream:
                        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                        status = child.wait()
                    value['completed_commands'].append(dict(command=command, pid=child.pid, started_unix=started,
                                                            ended_unix=time.time(), returncode=status, log=str(log), log_sha256=c.sha(log)))
                    c.save(cycle, value); assert status == 0, command
                report = c.read(audit)
                assert report['state'] == 'passed_all_updates_native_independent_patches_and_final_Adam'
                assert report['updates'] == (6 if args.scope == 'engineering' else 37120)
                value['rates'][str(rate)] = dict(manifest=str(manifest), manifest_sha256=c.sha(manifest),
                                                audit=str(audit), audit_sha256=c.sha(audit))
                c.save(cycle, value)
            value.update(state='finished_six_commands_pending_terminal_closure', ended_unix=time.time())
        except BaseException as error:
            value.update(state='failed', error=repr(error), ended_unix=time.time()); raise
        finally:
            c.save(cycle, value)
        return
    assert not closure.exists()
    start, run = c.read(launch), c.read(cycle)
    status = subprocess.run(['ps', '-p', str(start['pid']), '-o', 'pid,stat,args'], capture_output=True, text=True)
    assert status.returncode == 1 and run['pid'] == start['pid']
    assert run['state'] == 'finished_six_commands_pending_terminal_closure'
    assert start['sources'] == run['sources'] == c.sources()
    assert start['controller_sha256'] == run['controller_sha256'] == c.sha(script)
    assert len(run['completed_commands']) == 6 and all(row['returncode'] == 0 for row in run['completed_commands'])
    assert set(run['rates']) == {'10', '30', '50'}
    artifacts = {launch, cycle, Path(start['log'])}; targets = None; sampling = None; source_pngs = None; updates = 0
    for row in run['completed_commands']:
        assert c.sha(row['log']) == row['log_sha256']; artifacts.add(Path(row['log']))
    for rate, item in run['rates'].items():
        manifest, audit = Path(item['manifest']), Path(item['audit'])
        assert c.sha(manifest) == item['manifest_sha256'] and c.sha(audit) == item['audit_sha256']
        trained, report = c.read(manifest), c.read(audit)
        assert trained['state'] == 'finished' and report['state'] == 'passed_all_updates_native_independent_patches_and_final_Adam'
        assert trained['sources'] == report['sources'] == c.sources()
        assert report['manifest_sha256'] == c.sha(manifest)
        assert c.sha(trained['checkpoint']) == trained['checkpoint_sha256'] == report['checkpoint_sha256']
        assert c.sha(trained['raw_training_path']) == trained['raw_sha256'] == report['raw_sha256']
        rows = [json.loads(line) for line in Path(trained['raw_training_path']).read_text().splitlines()]
        current = [dict(step=row['step'], epoch=row['epoch'], views=row['views'], sampling=row['sampling'], target=row['target_tensor']) for row in rows]
        assert sampling is None or sampling == current; sampling = current
        assert targets is None or targets == report['target_hashes_by_step']; targets = report['target_hashes_by_step']
        assert source_pngs is None or source_pngs == trained['actual_source_images_sha256']; source_pngs = trained['actual_source_images_sha256']
        assert all(c.sha(path) == digest for path, digest in source_pngs.items())
        assert len(rows) == trained['optimizer_steps'] == report['updates'] == (6 if args.scope == 'engineering' else 37120)
        artifacts.update((manifest, audit, Path(trained['checkpoint']), Path(trained['raw_training_path'])))
        updates += len(rows)
    def mapped(path):
        if path.is_relative_to(ROOT):
            return str(path.relative_to(ROOT))
        assert path.is_relative_to(c.NATIVE) and path.relative_to(c.NATIVE).parts[0].startswith(prefix + '-cr')
        return 'data/engineering/' + prefix + '-native-transfer/' + str(path).lstrip('/')
    result = dict(state='closed_actual_terminal_all_three_rates_and_paired_training', scope=args.scope, actual_terminal=True,
                  checked_unix=time.time(), pid=start['pid'], sources=c.sources(), updates=updates, rates=3,
                  actual_source_views=len(source_pngs), artifacts_sha256={mapped(path): c.sha(path) for path in sorted(artifacts)},
                  native_to_local_artifacts={mapped(path): str(path) for path in sorted(artifacts)})
    with closure.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    (ROOT / 'data/provenance' / (prefix + '-transfer-files.txt')).write_text('\n'.join(str(path.relative_to(ROOT)) for path in sorted(artifacts) if path.is_relative_to(ROOT)) + '\n')
    (ROOT / 'data/provenance' / (prefix + '-native-transfer-files.txt')).write_text('\n'.join(str(path).lstrip('/') for path in sorted(artifacts) if not path.is_relative_to(ROOT)) + '\n')
    print(json.dumps(dict(state=result['state'], updates=updates, artifacts=len(artifacts))))


if __name__ == '__main__':
    main()
