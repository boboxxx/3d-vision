"""Independent fresh prediction/AP/erasure/received-identity native endpoint audit."""
import argparse
import math
from pathlib import Path
import time

from common import (ROOT, FINAL_PROTOCOL_SHA, check, current_sources, final_original_chain,
                    ordered_ids, read, save, sha256)


def validate_prediction(text, expected_count, erasure, allowed=('Car', 'Pedestrian', 'Cyclist')):
    lines = text.splitlines()
    check(len(lines) == expected_count, 'native prediction count differs')
    if erasure is not None:
        check(not lines, 'erasure must have empty native prediction')
    for line in lines:
        fields = line.split()
        check(len(fields) == 16 and fields[0] in allowed, 'native KITTI row schema')
        values = [float(v) for v in fields[1:]]
        check(all(math.isfinite(v) for v in values), 'finite KITTI values')
        check(0 <= values[-1] <= 1, 'native confidence range')
    return len(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    check(not args.output.exists(), 'retain previous endpoint audit')
    run = read(args.manifest)
    ids = ordered_ids()
    states = 484 if run['detector'] == 'liga' else 670
    check(run['state'] == 'finished' and run['ordered_ids'] == ids and run['frames_complete'] == 372
          and set(run['frames']) == set(ids) and run['seed'] == 17, 'full372 native endpoint')
    check(run['protocol_sha256'] == FINAL_PROTOCOL_SHA and run['final_chain'] == final_original_chain()
          and run['source_identities'] == current_sources(), 'whole final-source chain')
    check(run['detector_states_readonly'] == states and len(run['detector_initial_state_hashes']) == states
          and run['detector_initial_state_hashes'] == run['detector_final_state_hashes'], 'complete native readonly detector')
    output = Path(run['output_dir'])
    directory = output / 'data'
    check({p.stem for p in directory.glob('*.txt')} == set(ids), 'native KITTI full file set')
    cache = None
    if run['channel'] != 'clean_relay':
        check(sha256(run['cache_manifest_path']) == run['cache_manifest_sha256'], 'sealed received cache')
        cache = read(run['cache_manifest_path'])
        check(cache['channel'] == run['channel'] and cache['state'] == 'finished', 'cache condition')
    else:
        check(run['cache_manifest_sha256'] is None and run['cache_audit_sha256'] is None, 'clean scope')
    files = {}
    predictions = erasures = uses = 0
    root = Path('/mnt/d/paper6/data/kitti/training')
    for frame_id in ids:
        row = run['frames'][frame_id]
        path = directory / f'{frame_id}.txt'
        check(row['frame_id'] == frame_id and sha256(path) == row['prediction_sha256'], 'fresh native prediction')
        allowed = ('Car',) if run['detector'] == 'stereo_rcnn' else ('Car', 'Pedestrian', 'Cyclist')
        predictions += validate_prediction(path.read_text(), row['prediction_count'], row['erasure'], allowed)
        check(sha256(root / 'calib' / f'{frame_id}.txt') == row['calibration_sha256']
              and sha256(root / 'label_2' / f'{frame_id}.txt') == row['label_sha256'], 'fresh calibration/labels')
        for camera, digest in row['sensor_sha256'].items():
            check(sha256(root / camera / f'{frame_id}.png') == digest, 'fresh attempted sensor identity')
        if cache is not None:
            received = cache['frames'][frame_id]
            for key in ('received_file_sha256', 'received_tensors', 'erasure', 'accounting', 'sensor_sha256'):
                check(row[key] == received[key], 'same audited received data: ' + key)
            check(sha256(received['received_path']) == row['received_file_sha256'], 'fresh received file')
            uses += row['accounting']['total_uses']
        else:
            check(row['erasure'] is None and row['accounting'] is None, 'clean has no air resources')
        if row['erasure'] is not None:
            check(not row['detector_called'] and row['calls'] == {}, 'no detector/clean fallback on erasure')
            erasures += 1
        elif run['detector'] == 'liga':
            check(row['detector_called'] and row['calls'] == dict(image_backbone=2, feature_neck=2,
                                                                build_cost=1, head3D=1, forbidden=0), 'received native LIGA calls')
        else:
            count = row['native_3D_counts']
            check(row['detector_called'] and row['calls'] == dict(detector=1, image_backbone=2,
                                                                 dense_alignment=int(count['initial_solutions'] > 0)),
                  'received native StereoRCNN calls')
            check(count['dense_solutions'] == row['prediction_count'] and row['GT_placeholder_only'], 'complete native3D solver')
        files[frame_id] = dict(prediction_sha256=row['prediction_sha256'], label_sha256=row['label_sha256'],
                              calibration_sha256=row['calibration_sha256'], received_file_sha256=row['received_file_sha256'])
    check(sha256(output / 'metrics.json') == run['metrics_sha256'] and read(output / 'metrics.json') == run['metrics'], 'saved AP identity')
    # Independent invocation rereads all prediction and GT text with unchanged evaluator.
    from evaluate import metrics
    recomputed, _ = metrics(root.parent, ids, directory)
    for key in ('primary_Car_IoU_0_7_R40', 'secondary_Car_IoU_0_5'):
        check(recomputed[key] == run['metrics'][key], 'fresh full AP reduction mismatch')
    check(current_sources() == run['source_identities'], 'sources changed during fresh audit')
    result = dict(state='passed', detector=run['detector'], channel=run['channel'], frames=372,
                  manifest_sha256=sha256(args.manifest), predictions=predictions, erasures=erasures,
                  attempted_complex_uses=uses, detector_states_readonly=states, files=files,
                  cache_manifest_sha256=run['cache_manifest_sha256'], recomputed_metrics=recomputed,
                  checked_at_unix=time.time(), limitations=run['limitations'])
    save(args.output, result)
    print(__import__('json').dumps({k: v for k, v in result.items() if k not in ('files', 'recomputed_metrics')}))


if __name__ == '__main__':
    main()
