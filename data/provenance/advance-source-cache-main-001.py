"""Current-run dependency chain: native cache terminal -> full local proof -> AP.

Observe the existing native PID, never restart it. Dispatch main inference once
only after complete native closure and all22614 transferred records verify.
"""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'source-cache-main-to-inference-001'
REMOTE_ROOT = '/home/sheng/paper6'
HOST = 'sheng@100.94.183.27'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=10',
       '-o', 'ServerAliveCountMax=2', '-o', 'ControlPath=/private/tmp/paper6-sheng-01a10237.sock', HOST]
RSYNC_SSH = shlex.join(SSH[:-1])


def read(path):
    return json.loads(Path(path).read_text())


def main():
    state_path = ROOT / 'data/runs' / (PREFIX + '.json')
    assert not state_path.exists()
    assert read(ROOT / 'data/provenance/srcnn-kitti-engineering-seed17-002-local-verification.json')['state'] == 'passed_all18_transferred_SRCNN_training_records_and_terminal_artifacts'
    state = dict(state='running', pid=os.getpid(), started_unix=time.time(), phase='observe_existing_cache_PID',
                 existing_native_cache_PID=265830, completed_stages=[])
    def save():
        temp = state_path.with_suffix('.tmp')
        temp.write_text(json.dumps(state, indent=2, allow_nan=False)); temp.replace(state_path)
    def stage(name, command):
        state.update(phase=name, phase_started_unix=time.time()); save()
        log = ROOT / 'logs' / (PREFIX + '-' + name + '.log')
        print(json.dumps(dict(phase=name)), flush=True)
        with log.open('x') as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        assert result.returncode == 0, (name, result.returncode)
        state['completed_stages'].append(dict(name=name, finished_unix=time.time(), log=str(log)))
        save(); return log
    save()
    try:
        # Actual PID/terminal checks, not a manifest-only wait. Observation errors
        # do not imply terminal state and never authorize restarting the cache.
        failures = 0
        while True:
            code = ('import json,subprocess,time; '
                    'd=json.load(open("data/runs/original-source-cache-main-001-cycle.json")); '
                    'p=subprocess.run(["ps","-p",str(d["pid"]),"-o","pid,stat"],capture_output=True,text=True); '
                    'print(json.dumps(dict(checked_unix=time.time(),state=d["state"],pid=d["pid"],'
                    'ps_returncode=p.returncode,actual_ps=p.stdout,commands=len(d["completed_commands"]),error=d.get("error"))))')
            command = SSH + ['cd ' + shlex.quote(REMOTE_ROOT) + ' && python3 -c ' + shlex.quote(code)]
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=40)
                assert result.returncode == 0, result.stderr
                snapshot = json.loads(result.stdout)
                assert snapshot['pid'] == 265830
                failures = 0
            except (subprocess.TimeoutExpired, AssertionError, json.JSONDecodeError) as error:
                failures += 1
                state.update(last_observation_error=repr(error), observation_failures=failures, last_poll_unix=time.time())
                save(); assert failures < 3, 'SSH observation failed; native state unknown, no restart'
                time.sleep(30); continue
            state.update(last_actual_native_snapshot=snapshot, last_poll_unix=time.time()); save()
            if snapshot['state'] == 'finished_eight_commands_pending_terminal_closure':
                assert snapshot['commands'] == 8
                if snapshot['ps_returncode'] == 1:
                    break
            else:
                assert snapshot['state'] == 'running' and snapshot['ps_returncode'] == 0, snapshot
            time.sleep(30)
        remote = ('cd ' + shlex.quote(REMOTE_ROOT) + ' && source scripts/sheng_env.sh && '
                  'export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 && '
                  'python data/provenance/source-cache-cycle-001.py --mode close --scope main')
        stage('native_terminal_closure', SSH + [remote])
        closure_path = ROOT / 'data/provenance/original-source-cache-main-001-closure.json'
        assert not closure_path.exists()
        stage('transfer_native_closure', ['rsync', '-az', '-e', RSYNC_SSH,
                                          HOST + ':' + REMOTE_ROOT + '/data/provenance/' + closure_path.name,
                                          str(closure_path.parent) + '/'])
        closure = read(closure_path)
        assert closure['state'] == 'closed_actual_terminal_all_native_artifacts' and closure['actual_terminal']
        assert closure['pairs'] == 22614 and len(closure['transferred_files']) == 19
        paths = sorted(closure['transferred_files'])
        for path in paths:
            value = Path(path)
            assert value.is_absolute() and '..' not in value.parts
            assert value.is_relative_to(REMOTE_ROOT) or value.is_relative_to('/mnt/d/paper6/runs/original-source-cache-main-001')
        listing = ROOT / 'data/provenance/original-source-cache-main-001-driver-transfer-files-001.txt'
        with listing.open('x') as stream:
            stream.write('\n'.join(path.lstrip('/') for path in paths) + '\n')
        mirror = ROOT / 'data/engineering/original-source-cache-main-001-transfer'
        mirror.mkdir()
        stage('transfer_all19_native_records', ['rsync', '-az', '--files-from=' + str(listing), '-e', RSYNC_SSH,
                                                HOST + ':/', str(mirror) + '/'])
        stage('local_all22614_record_verification', [sys.executable, 'data/provenance/verify-source-cache-main-local-001.py'])
        proof_path = ROOT / 'data/provenance/original-source-cache-main-001-local-verification.json'
        assert read(proof_path)['state'] == 'passed_all22614_transferred_pair_records_and_native_audit_metadata'
        stage('upload_local_proof', ['rsync', '-az', '-e', RSYNC_SSH, str(proof_path),
                                     HOST + ':' + REMOTE_ROOT + '/data/provenance/'])
        remote = ('cd ' + shlex.quote(REMOTE_ROOT) + ' && source scripts/sheng_env.sh && '
                  'export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 WANDB_MODE=disabled && '
                  'python data/provenance/source-cache-inference-cycle-001.py --mode launch --scope main')
        log = stage('launch12_main_detector_endpoints_once', SSH + [remote])
        launched = json.loads(log.read_text())
        assert launched['state'] == 'launched_once' and launched['prefix'] == 'source-cache-inference-main-001'
        state.update(state='finished_dependency_chain_main_AP_dispatched', main_native_PID=launched['pid'],
                     main_native_launch=launched, ended_unix=time.time()); save()
        print(json.dumps(dict(state=state['state'], main_native_PID=state['main_native_PID'])), flush=True)
    except BaseException as error:
        state.update(state='failed', error=repr(error), ended_unix=time.time()); save(); raise


if __name__ == '__main__':
    main()
