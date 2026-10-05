"""Two received caches, six fixed native endpoints and paired independent closure."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from common import ROOT, HERE, check, current_sources, final_original_chain, ordered_ids, read, save, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    check(re.fullmatch(r'[a-z0-9][a-z0-9-]{3,60}', args.prefix), 'safe unique prefix')
    os.chdir(ROOT)
    path = ROOT / f'data/runs/{args.prefix}-cycle.json'
    check(not path.exists(), 'retain previous cycle')
    chain, sources = final_original_chain(), current_sources()
    result = dict(state='running', pid=os.getpid(), started_at_unix=time.time(), source_identities=sources,
                  final_chain=chain, completed_commands=[], endpoints={}, prefix=args.prefix)

    def invoke(label, command):
        check(current_sources() == sources, 'frozen execution sources changed')
        log = ROOT / f'logs/{args.prefix}-{label}.log'
        result['current_command'] = label
        save(path, result)
        with log.open('x') as stream:
            subprocess.run([sys.executable, *map(str, command)], cwd=ROOT, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        check(current_sources() == sources, 'frozen execution sources changed')
        result['completed_commands'].append(dict(label=label, log_sha256=sha256(log)))
        save(path, result)

    save(path, result)
    try:
        invoke('received-cache', [HERE / 'cache.py', '--prefix', args.prefix + '-cache'])
        for channel in ('identity', 'awgn'):
            cache = ROOT / f'data/runs/{args.prefix}-cache-{channel}.json'
            audit = cache.with_name(cache.stem + '-audit.json')
            invoke(channel + '-cache-audit', [HERE / 'audit_cache.py', '--manifest', cache, '--output', audit])
        for detector in ('stereo_rcnn', 'liga'):
            for channel in ('clean_relay', 'identity', 'awgn'):
                rid = f'{args.prefix}-{detector.replace("_", "-")}-{channel.replace("_", "-")}'
                manifest = ROOT / f'data/runs/{rid}.json'
                audit = manifest.with_name(manifest.stem + '-audit.json')
                command = [HERE / 'evaluate.py', '--detector', detector, '--channel', channel, '--run-id', rid]
                if channel != 'clean_relay':
                    cache = ROOT / f'data/runs/{args.prefix}-cache-{channel}.json'
                    command.extend(['--cache-manifest', cache, '--cache-audit', cache.with_name(cache.stem + '-audit.json')])
                while True:
                    free = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                                                    '--format=csv,noheader,nounits'], text=True).strip().splitlines()
                    check(len(free) == 1, 'single GPU')
                    if int(free[0]) >= 12288:
                        break
                    result.update(state='waiting_physical_GPU_margin', last_free_MiB=int(free[0]))
                    save(path, result)
                    time.sleep(15)
                result['state'] = 'running'
                invoke(rid, command)
                invoke(rid + '-audit', [HERE / 'audit_endpoint.py', '--manifest', manifest, '--output', audit])
                run, verified = read(manifest), read(audit)
                check(verified['state'] == 'passed' and verified['frames'] == 372, 'full endpoint audit')
                result['endpoints'][detector + ':' + channel] = dict(manifest_path=str(manifest),
                    manifest_sha256=sha256(manifest), audit_path=str(audit), audit_sha256=sha256(audit),
                    metrics=verified['recomputed_metrics'], cache_manifest_sha256=run['cache_manifest_sha256'])
                save(path, result)
        # Same exact received tensors for both author detectors, all372 including erasures.
        for channel in ('clean_relay', 'identity', 'awgn'):
            a, b = [read(result['endpoints'][d + ':' + channel]['manifest_path']) for d in ('stereo_rcnn', 'liga')]
            check(a['cache_manifest_sha256'] == b['cache_manifest_sha256'], 'paired received cache identity')
            for frame in ordered_ids():
                for key in ('received_tensors', 'received_file_sha256', 'erasure', 'accounting', 'sensor_sha256'):
                    check(a['frames'][frame][key] == b['frames'][frame][key], 'paired received frame identity: ' + key)
        result.update(state='finished_all_six_audits_and_paired_inputs_passed', ended_at_unix=time.time(),
                      limitations='Full372 exploratory declared original variant, author-pretraining overlap, unmatched rates/energy/exposures; not fair superiority or project completion')
        save(path, result)
        print(json.dumps(dict(state=result['state'], endpoints=len(result['endpoints']))), flush=True)
    except BaseException as exc:
        result.update(state='failed', exception=repr(exc), ended_at_unix=time.time())
        save(path, result)
        raise


if __name__ == '__main__':
    main()
