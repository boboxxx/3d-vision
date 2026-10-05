"""Terminal all-file/source/paired-received closure of the six native endpoints."""
import argparse
from pathlib import Path
import time

from common import ROOT, check, current_sources, final_original_chain, ordered_ids, read, save, sha256, terminal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    path = ROOT / f'data/runs/{args.prefix}-cycle.json'
    closure = ROOT / f'data/provenance/{args.prefix}-closure.json'
    check(not closure.exists(), 'retain previous closure')
    cycle = read(path)
    check(cycle['state'] == 'finished_all_six_audits_and_paired_inputs_passed' and terminal(cycle['pid']), 'complete terminal cycle')
    check(cycle['source_identities'] == current_sources() and cycle['final_chain'] == final_original_chain(), 'full final source chain')
    check(set(cycle['endpoints']) == {d + ':' + c for d in ('stereo_rcnn', 'liga')
                                    for c in ('clean_relay', 'identity', 'awgn')}, 'six fixed endpoints')
    artifacts = {str(path.relative_to(ROOT)): sha256(path)}
    ids = ordered_ids()
    for command in cycle['completed_commands']:
        log = ROOT / f"logs/{args.prefix}-{command['label']}.log"
        check(sha256(log) == command['log_sha256'], 'retained command log')
        artifacts[str(log.relative_to(ROOT))] = sha256(log)
    for channel in ('identity', 'awgn'):
        cache_path = ROOT / f'data/runs/{args.prefix}-cache-{channel}.json'
        audit_path = cache_path.with_name(cache_path.stem + '-audit.json')
        cache, audit = read(cache_path), read(audit_path)
        check(audit['state'] == 'passed' and audit['frames'] == cache['frames_complete'] == 372
              and audit['manifest_sha256'] == sha256(cache_path) and cache['source_identities'] == current_sources(), 'full cache audit identity')
        for frame in ids:
            row = cache['frames'][frame]
            check(sha256(row['received_path']) == row['received_file_sha256'] == audit['files'][frame]['received_file_sha256'], 'fresh received arrays')
        artifacts[str(cache_path.relative_to(ROOT))] = sha256(cache_path)
        artifacts[str(audit_path.relative_to(ROOT))] = sha256(audit_path)
    root = Path('/mnt/d/paper6/data/kitti/training')
    endpoints = {}
    for key, item in cycle['endpoints'].items():
        manifest, audit_path = Path(item['manifest_path']), Path(item['audit_path'])
        run, audit = read(manifest), read(audit_path)
        check(sha256(manifest) == item['manifest_sha256'] == audit['manifest_sha256']
              and sha256(audit_path) == item['audit_sha256'], 'complete endpoint/audit identity')
        check(audit['state'] == 'passed' and audit['frames'] == 372 and run['state'] == 'finished'
              and audit['recomputed_metrics'] == item['metrics'] == run['metrics'], 'full endpoint/AP closure')
        metrics = Path(run['output_dir']) / 'metrics.json'
        check(sha256(metrics) == run['metrics_sha256'], 'saved native AP identity')
        for frame in ids:
            files = audit['files'][frame]
            for p, name in ((Path(run['output_dir']) / 'data' / f'{frame}.txt', 'prediction_sha256'),
                            (root / 'label_2' / f'{frame}.txt', 'label_sha256'),
                            (root / 'calib' / f'{frame}.txt', 'calibration_sha256')):
                check(sha256(p) == files[name], 'fresh prediction/GT/calibration closure')
        artifacts[str(manifest.relative_to(ROOT))] = sha256(manifest)
        artifacts[str(audit_path.relative_to(ROOT))] = sha256(audit_path)
        endpoints[key] = run
    for channel in ('clean_relay', 'identity', 'awgn'):
        left, right = [endpoints[d + ':' + channel] for d in ('stereo_rcnn', 'liga')]
        check(left['cache_manifest_sha256'] == right['cache_manifest_sha256'], 'same air attempt for both native detectors')
        for frame in ids:
            for key in ('received_file_sha256', 'received_tensors', 'erasure', 'accounting', 'sensor_sha256'):
                check(left['frames'][frame][key] == right['frames'][frame][key], 'paired received image provenance')
    save(closure, dict(state='closed_all_audits_passed', prefix=args.prefix, checked_at_unix=time.time(),
                       artifacts_sha256=artifacts, endpoints=cycle['endpoints'], sources=cycle['source_identities'],
                       final_chain=cycle['final_chain'], limitations=cycle['limitations']))
    print('All six native endpoints and shared received caches terminally closed.')


if __name__ == '__main__':
    main()
