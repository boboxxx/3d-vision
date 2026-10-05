"""After prior engineering closes, validate two final receiver implementations.

One-frame fresh768 engineering only. No formal cache/evaluation starts here.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / 'experiments/original-detector-integration/final-code'
sys.path.insert(0, str(CODE))
from common import NATIVE_PROBES, check, current_sources, read, save, sha256, terminal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--identity', type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    identity = read(args.identity)
    path = ROOT / 'data/runs/final-receiver-queue-001.json'
    check(not path.exists(), 'retain previous final receiver queue')

    def frozen():
        check({k:v['sha256'] for k,v in current_sources().items()} == identity['sources_sha256'], 'frozen final evaluation source changed')
        check(sha256(Path(__file__).resolve()) == identity['queue_sha256'], 'queue implementation changed')
        check(sha256(ROOT / identity['CPU_evidence_path']) == identity['CPU_evidence_sha256'], 'CPU evidence changed')

    frozen()
    result = dict(state='waiting_previous_native_engineering_closure', pid=os.getpid(),
                  started_at_unix=time.time(), identity_sha256=sha256(args.identity), completed_commands=[],
                  scope='two real one-frame fresh768 final-receiver engineering probes only, no AP/formal baseline')
    save(path, result)
    try:
        while True:
            previous = read(ROOT / 'data/runs/native-validation-queue-001.json')
            check(previous['state'] != 'failed', 'previous native queue failed; preserve failure and stop')
            if previous['state'] == 'finished_native_engineering_closed' and terminal(previous['pid']):
                break
            result.update(last_previous_queue_state=previous['state'], last_checked_at_unix=time.time())
            save(path, result)
            time.sleep(15)
        for detector, name in NATIVE_PROBES.items():
            output = ROOT / 'data/engineering' / name
            check(not output.exists(), 'retain previous native probe')
            while True:
                memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                                                  '--format=csv,noheader,nounits'], text=True).strip().splitlines()
                check(len(memory) == 1, 'single GPU')
                if int(memory[0]) >= 12288:
                    break
                result.update(state='waiting_physical_GPU_margin', last_free_MiB=int(memory[0]))
                save(path, result)
                time.sleep(15)
            frozen()
            result.update(state='running', detector=detector)
            save(path, result)
            log = ROOT / 'logs' / (output.stem + '.log')
            with log.open('x') as stream:
                subprocess.run([sys.executable, str(CODE / 'probe_receivers.py'), '--detector', detector,
                                '--output', str(output)], cwd=ROOT, stdout=stream,
                               stderr=subprocess.STDOUT, check=True)
            probe = read(output)
            check(probe['state'] == 'passed', 'new native receiver engineering failed')
            result['completed_commands'].append(dict(detector=detector, probe_path=str(output),
                                                      probe_sha256=sha256(output), log_sha256=sha256(log)))
            frozen()
            save(path, result)
        a, b = [read(ROOT / 'data/engineering' / NATIVE_PROBES[d]) for d in ('stereo_rcnn', 'liga')]
        for left, right in zip(a['conditions'], b['conditions']):
            check(all(left[k] == right[k] for k in ('channel', 'received_tensors', 'accounting', 'erasure')), 'paired real received inputs')
        result.update(state='finished_paired_native_receiver_engineering', ended_at_unix=time.time(), formal_baseline_started=False)
        save(path, result)
    except BaseException as exc:
        result.update(state='failed', exception=repr(exc), ended_at_unix=time.time())
        save(path, result)
        raise


if __name__ == '__main__':
    main()
