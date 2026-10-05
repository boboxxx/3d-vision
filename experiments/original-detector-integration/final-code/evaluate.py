"""Full372 fixed native author receiver of audited final received RGB."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import torch
from common import (ROOT, HERE, FINAL_PROTOCOL_SHA, check, current_sources, final_original_chain,
                    load_received, ordered_ids, read, save, sha256, tensor_identity)
from geocomm.pooling_diagnostic import state_hashes
from receivers import LigaReceiver, StereoReceiver


def metrics(root, ids, directory):
    # All inference outputs are already fixed before labels are read here.
    sys.path.insert(0, str(ROOT / 'third_party/LIGA-Stereo/liga/datasets/kitti'))
    from kitti_object_eval_python import kitti_common
    from kitti_object_eval_python.eval import get_official_eval_result, do_eval
    gt = kitti_common.get_label_annos(root / 'training/label_2', [int(v) for v in ids])
    dt = kitti_common.get_label_annos(directory, [int(v) for v in ids])
    text, strict = get_official_eval_result(gt, dt, ['Car'])
    arrays = do_eval(gt, dt, [0], np.full((1, 3, 1), .5), compute_aos=True)
    relaxed = {}
    for name, index in (('bbox', 0), ('bev', 1), ('3d', 2), ('aos', 3)):
        for recall, offset in (('R11', 0), ('R40', 4)):
            if arrays[index + offset] is not None:
                relaxed[name + '_' + recall] = arrays[index + offset][0, :, 0].tolist()
    result = dict(primary_Car_IoU_0_7_R40={k: float(v) for k, v in strict.items()},
                  secondary_Car_IoU_0_5=relaxed, difficulty_order=['easy', 'moderate', 'hard'])
    check(np.isfinite(list(strict.values()) + [v for row in relaxed.values() for v in row]).all(), 'finite AP')
    return result, text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector', choices=('liga', 'stereo_rcnn'), required=True)
    parser.add_argument('--channel', choices=('clean_relay', 'identity', 'awgn'), required=True)
    parser.add_argument('--cache-manifest', type=Path)
    parser.add_argument('--cache-audit', type=Path)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    check(re.fullmatch(r'[a-z0-9][a-z0-9-]{3,90}', args.run_id), 'safe unique run ID')
    check((args.cache_manifest is None) == (args.channel == 'clean_relay'), 'fixed clean/cache condition')
    chain = final_original_chain()
    sources = current_sources()
    ids = ordered_ids()
    root = Path('/mnt/d/paper6/data/kitti')
    directory = Path('/mnt/d/paper6/runs') / args.run_id
    manifest = ROOT / f'data/runs/{args.run_id}.json'
    check(not directory.exists() and not manifest.exists(), 'retain previous endpoint')
    cache, cache_audit_sha = None, None
    if args.cache_manifest is not None:
        check(args.cache_audit is not None, 'complete independent cache audit required')
        cache, audit = read(args.cache_manifest), read(args.cache_audit)
        check(cache['state'] == 'finished' and cache['channel'] == args.channel and
              cache['source_identities'] == sources and cache['final_chain'] == chain, 'cache source/condition')
        check(audit['state'] == 'passed' and audit['frames'] == 372 and audit['channel'] == args.channel
              and audit['manifest_sha256'] == sha256(args.cache_manifest)
              and audit['checkpoint_sha256'] == chain['checkpoint_sha256'], 'sealed cache audit')
        cache_audit_sha = sha256(args.cache_audit)
    usage = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                                     '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    check(len(usage) == 1, 'single GPU')
    free = int(usage[0]) * 2**20
    check(free >= 12 * 2**30, 'physical free12GiB, no other job interruption')
    torch.set_num_threads(2)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    np.random.seed(17)
    directory.mkdir(parents=True)
    predictions_dir = directory / 'data'
    predictions_dir.mkdir()
    record = dict(state='running', run_id=args.run_id, detector=args.detector, channel=args.channel,
                  seed=17, ordered_ids=ids, frames_complete=0, frames={}, source_identities=sources,
                  final_chain=chain, protocol_sha256=FINAL_PROTOCOL_SHA, output_dir=str(directory),
                  cache_manifest_path=str(args.cache_manifest) if cache is not None else None,
                  cache_manifest_sha256=sha256(args.cache_manifest) if cache is not None else None,
                  cache_audit_sha256=cache_audit_sha, NVIDIA_free_before_bytes=free, started_at_unix=time.time())
    save(manifest, record)
    receiver = None
    try:
        receiver = (LigaReceiver if args.detector == 'liga' else StereoReceiver)()
        check(all(not module.training for module in receiver.model.modules()), 'all receiver modules eval')
        record.update(detector_checkpoint_sha256=receiver.checkpoint_sha,
                      detector_initial_state_hashes=receiver.initial)
        clean_dataset = None
        if cache is None:
            sys.path.insert(0, str(ROOT / 'reproduction/cao2025'))
            from data import StereoRGB
            clean_dataset = StereoRGB(root, ROOT / 'data/engineering/cao2025-roi-holdout-001.jsonl',
                                      ROOT / 'data/engineering/cao2025-roi-holdout-audit-001.json',
                                      ROOT / 'data/internal-tuning-fold-001.json', 'geocomm_tune_holdout')
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for index, frame_id in enumerate(ids):
                if cache is not None:
                    outputs, row = load_received(args.cache_manifest, frame_id)
                    provenance = dict(received_file_sha256=row['received_file_sha256'],
                                      received_tensors=row['received_tensors'], erasure=row['erasure'],
                                      accounting=row['accounting'], sensor_sha256=row['sensor_sha256'])
                else:
                    frame = clean_dataset[index]
                    check(frame['frame_id'] == frame_id, 'clean ordered sensor')
                    outputs = (frame['left'], frame['right'])
                    provenance = dict(received_file_sha256=None, received_tensors=[tensor_identity(v) for v in outputs],
                                      erasure=None, accounting=None,
                                      sensor_sha256={v['camera']: v['image_sha256'] for v in clean_dataset.rows[index]['views']})
                prediction_path = predictions_dir / f'{frame_id}.txt'
                if outputs is None:
                    prediction_path.write_text('')
                    endpoint = dict(prediction_count=0, calls={}, detector_called=False)
                else:
                    endpoint = receiver.predict(outputs, frame_id, root / 'training/calib' / f'{frame_id}.txt', predictions_dir)
                    endpoint['detector_called'] = True
                torch.cuda.synchronize()
                lines = prediction_path.read_text().splitlines()
                check(len(lines) == endpoint['prediction_count'], 'full native writer count')
                allowed = ('Car',) if args.detector == 'stereo_rcnn' else ('Car', 'Pedestrian', 'Cyclist')
                check(all(len(line.split()) == 16 and line.split()[0] in allowed and
                          np.isfinite([float(v) for v in line.split()[1:]]).all() for line in lines), 'finite native KITTI rows')
                check(state_hashes(receiver.model) == receiver.initial, 'all native receiver states readonly')
                check(torch.cuda.max_memory_reserved() + 2 * 2**30 <= free, 'measured physical memory margin')
                record['frames'][frame_id] = dict(frame_id=frame_id, **provenance, **endpoint,
                                                 prediction_sha256=sha256(prediction_path),
                                                 calibration_sha256=sha256(root / 'training/calib' / f'{frame_id}.txt'))
                record['frames_complete'] = index + 1
                save(manifest, record)
                if (index + 1) % 20 == 0:
                    print(json.dumps(dict(detector=args.detector, channel=args.channel, frames=index + 1)), flush=True)
                del outputs
        check({p.stem for p in predictions_dir.glob('*.txt')} == set(ids), 'complete372 prediction set')
        # The inference loop has ended before label access/AP reduction.
        ap, text = metrics(root, ids, predictions_dir)
        save(directory / 'metrics.json', ap)
        (directory / 'evaluator.txt').write_text(text)
        for frame_id in ids:
            record['frames'][frame_id]['label_sha256'] = sha256(root / 'training/label_2' / f'{frame_id}.txt')
        check(current_sources() == sources and (cache is None or sha256(args.cache_manifest) == record['cache_manifest_sha256']),
              'source/cache identity changed')
        record.update(state='finished', metrics=ap, metrics_sha256=sha256(directory / 'metrics.json'),
                      detector_states_readonly=receiver.states, detector_final_state_hashes=state_hashes(receiver.model),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(), ended_at_unix=time.time(),
                      limitations='Internal372, author-pretraining overlap, declared original variant; unmatched rate/exposure, no fair gain/mainval/novelty/latency claim')
        save(manifest, record)
    except BaseException as exc:
        record.update(state='failed', exception=repr(exc), ended_at_unix=time.time())
        save(manifest, record)
        raise
    finally:
        if receiver is not None:
            receiver.close()


if __name__ == '__main__':
    main()
