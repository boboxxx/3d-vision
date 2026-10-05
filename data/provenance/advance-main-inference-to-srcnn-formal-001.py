"""Existing main PID -> complete native/local proof -> once-only SRCNN formal.

This is the current experiment's dependency chain, not a recurring task. It
never restarts a failed endpoint or native cycle, and preserves failed stages.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
REMOTE_ROOT = '/home/sheng/paper6'
HOST = 'sheng@100.94.183.27'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=10',
       '-o', 'ServerAliveCountMax=2', '-o', 'ControlPath=/private/tmp/paper6-sheng-01a10237.sock', HOST]
RSYNC_SSH = shlex.join(SSH[:-1])
PREFIX = 'main-inference-to-srcnn-formal-001'
MAIN = 'source-cache-inference-main-001'
CONDITIONS = [codec + '-cr' + str(rate) for codec in ('jpeg', 'jpeg2000') for rate in (10, 30, 50)]


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return h.hexdigest()


def transfer_plan(closure, scope):
    prefix = 'source-cache-inference-' + scope + '-001'
    assert closure['state'] == 'closed_all12_endpoints_actual_terminal_and_shared_received_inputs'
    assert closure['actual_terminal'] and closure['scope'] == scope
    assert closure['frames'] == (24 if scope == 'engineering' else 45228)
    assert closure['shared_pairs'] == closure['frames'] // 2
    assert len(closure['artifacts_sha256']) == (75 if scope == 'engineering' else 45279)
    assert set(closure['artifacts_sha256']) == set(closure['native_to_local_artifacts'])
    roots, native = [], []
    directories = {prefix + '-' + condition + '-' + detector for condition in CONDITIONS for detector in ('stereo_rcnn', 'liga')}
    for local, value in closure['native_to_local_artifacts'].items():
        path = Path(value); assert path.is_absolute() and '..' not in path.parts
        if path.is_relative_to(REMOTE_ROOT):
            rel = path.relative_to(REMOTE_ROOT)
            assert rel.parts[:2] in (('data', 'runs'), ('data', 'provenance')) or rel.parts[0] == 'logs'
            assert rel.name.startswith(prefix) and local == str(rel)
            roots.append(str(rel))
        else:
            assert path.is_relative_to('/mnt/d/paper6/runs')
            rel = path.relative_to('/mnt/d/paper6/runs'); assert rel.parts[0] in directories
            assert (len(rel.parts) == 3 and rel.parts[1] == 'data' and path.suffix == '.txt'
                    and len(path.stem) == 6 and path.stem.isdecimal()) or (
                scope == 'main' and len(rel.parts) == 2 and path.name in ('metrics.json', 'evaluator.txt'))
            expected = 'data/engineering/' + prefix + '-native-transfer/' + str(path).lstrip('/')
            assert local == expected; native.append(str(path).lstrip('/'))
    assert len(roots) == 51 and len(native) == (24 if scope == 'engineering' else 45228 + 24)
    return sorted(roots), sorted(native)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', choices=('check-plan', 'run'), required=True)
    args = parser.parse_args()
    if args.mode == 'check-plan':
        path = ROOT / 'data/provenance/source-cache-inference-engineering-001-closure.json'
        closure = read(path); roots, native = transfer_plan(closure, 'engineering')
        for relative, digest in closure['artifacts_sha256'].items(): assert sha(ROOT / relative) == digest
        print(json.dumps(dict(state='passed_existing_complete_engineering_transfer_plan', artifacts=75,
                              root_artifacts=len(roots), native_artifacts=len(native)))); return
    state_path = ROOT / 'data/runs' / (PREFIX + '.json'); assert not state_path.exists()
    complete = ROOT / 'data/provenance/srcnn-kitti-engineering-seed17-002-complete-local-003.json'
    proof = read(complete)
    assert proof['state'] == 'passed_all18_transferred_SRCNN_training_records_and_terminal_artifacts'
    assert proof['verifier_sha256'] == sha(ROOT / 'data/provenance/verify-srcnn-kitti-complete-local-003.py')
    state = dict(state='running', pid=os.getpid(), started_unix=time.time(), existing_main_native_PID=269262,
                 driver_sha256=sha(Path(__file__)), engineering_complete_proof_sha256=sha(complete),
                 phase='observe_existing_main_PID', completed_stages=[])
    def save():
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2, allow_nan=False)); temporary.replace(state_path)
    def stage(name, command):
        state.update(phase=name, phase_started_unix=time.time()); save()
        log = ROOT / 'logs' / (PREFIX + '-' + name + '.log')
        print(json.dumps(dict(phase=name)), flush=True)
        with log.open('x') as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        state['completed_stages'].append(dict(name=name, returncode=result.returncode, finished_unix=time.time(),
                                              log=str(log), log_sha256=sha(log))); save()
        assert result.returncode == 0, (name, result.returncode)
        return log
    save()
    try:
        failures = 0
        while True:
            code = ('import json,subprocess,time; '
                    'j=json.load(open("data/runs/source-cache-inference-main-001-cycle.json")); '
                    'p=subprocess.run(["ps","-p",str(j["pid"]),"-o","pid,stat,args"],capture_output=True,text=True); '
                    'print(json.dumps(dict(checked_unix=time.time(),state=j["state"],pid=j["pid"],'
                    'ps_returncode=p.returncode,actual_ps=p.stdout,commands=len(j["completed_commands"]),'
                    'returncodes=[v["returncode"] for v in j["completed_commands"]],'
                    'endpoints=len(j["endpoints"]),error=j.get("error"))))')
            command = SSH + ['cd ' + shlex.quote(REMOTE_ROOT) + ' && python3 -c ' + shlex.quote(code)]
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=40)
                assert result.returncode == 0, result.stderr
                snapshot = json.loads(result.stdout); assert snapshot['pid'] == 269262
                failures = 0
            except (subprocess.TimeoutExpired, AssertionError, json.JSONDecodeError) as error:
                failures += 1; state.update(last_observation_error=repr(error), observation_failures=failures,
                                           last_poll_unix=time.time()); save()
                assert failures < 3, 'Native state unknown; do not restart or dispatch SRCNN'
                time.sleep(30); continue
            state.update(last_actual_native_snapshot=snapshot, last_poll_unix=time.time()); save()
            assert snapshot['error'] is None and all(v == 0 for v in snapshot['returncodes']), snapshot
            if snapshot['state'] == 'finished12_native_endpoints12_audits_pending_terminal_closure':
                assert snapshot['commands'] == 24 and snapshot['endpoints'] == 12
                if snapshot['ps_returncode'] == 1: break
                assert snapshot['ps_returncode'] == 0, snapshot
            else:
                assert snapshot['state'] == 'running' and snapshot['ps_returncode'] == 0, snapshot
            time.sleep(30)
        remote = ('cd ' + shlex.quote(REMOTE_ROOT) + ' && source scripts/sheng_env.sh && '
                  'export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 && '
                  'python data/provenance/source-cache-inference-cycle-001.py --mode close --scope main')
        stage('native_actual_terminal_closure', SSH + [remote])
        closure_path = ROOT / 'data/provenance' / (MAIN + '-closure.json'); assert not closure_path.exists()
        stage('transfer_native_closure', ['rsync', '-az', '-e', RSYNC_SSH,
                                         HOST + ':' + REMOTE_ROOT + '/data/provenance/' + closure_path.name,
                                         str(closure_path.parent) + '/'])
        closure = read(closure_path); roots, native = transfer_plan(closure, 'main')
        root_list = ROOT / 'data/provenance' / (MAIN + '-dependency-root-transfer-001.txt')
        native_list = ROOT / 'data/provenance' / (MAIN + '-dependency-native-transfer-001.txt')
        for path, values in ((root_list, roots), (native_list, native)):
            with path.open('x') as stream: stream.write('\n'.join(values) + '\n')
        mirror = ROOT / 'data/engineering' / (MAIN + '-native-transfer'); mirror.mkdir()
        stage('transfer_all_root_native_reports_and_logs', ['rsync', '-az', '--files-from=' + str(root_list),
              '-e', RSYNC_SSH, HOST + ':' + REMOTE_ROOT + '/', str(ROOT) + '/'])
        stage('transfer_all45228_predictions_and24_metric_files', ['rsync', '-az', '--files-from=' + str(native_list),
              '-e', RSYNC_SSH, HOST + ':/', str(mirror) + '/'])
        stage('local_all45228_prediction_verification', [sys.executable, 'data/provenance/verify-source-cache-inference-main-local-001.py'])
        local_proof = ROOT / 'data/provenance' / (MAIN + '-local-verification.json'); passed = read(local_proof)
        assert passed['state'] == 'passed_all45228_transferred_native_inference_records_predictions_and_metrics'
        assert passed['closure_sha256'] == sha(closure_path)
        stage('upload_complete_local_main_proof', ['rsync', '-az', '-e', RSYNC_SSH, str(local_proof),
                                                   HOST + ':' + REMOTE_ROOT + '/data/provenance/'])
        # Extra compute-process gate supplements the frozen controller's free-memory check.
        gpu_code = ('import subprocess; p=subprocess.run(["nvidia-smi","--query-compute-apps=pid",'
                    '"--format=csv,noheader,nounits"],capture_output=True,text=True,check=True); '
                    'assert not p.stdout.strip(), "GPU processes still present: "+p.stdout; print("no_active_GPU_compute_processes")')
        stage('verify_no_competing_GPU_process', SSH + ['python3 -c ' + shlex.quote(gpu_code)])
        remote = ('cd ' + shlex.quote(REMOTE_ROOT) + ' && source scripts/sheng_env.sh && '
                  'export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 WANDB_MODE=disabled && '
                  'python data/provenance/srcnn-kitti-cycle-002.py --mode launch --scope formal')
        log = stage('launch_formal_SRCNN_three_rates_once', SSH + [remote]); launched = read(log)
        assert launched['state'] == 'launched_once' and launched['scope'] == 'formal'
        assert launched['prefix'] == 'srcnn-kitti-formal-seed17-002'
        state.update(state='finished_complete_main_proof_and_SRCNN_formal_dispatched',
                     formal_native_PID=launched['pid'], formal_native_launch=launched, ended_unix=time.time()); save()
        print(json.dumps(dict(state=state['state'], formal_native_PID=launched['pid'])), flush=True)
    except BaseException as error:
        state.update(state='failed', error=repr(error), ended_unix=time.time()); save(); raise


if __name__ == '__main__': main()
