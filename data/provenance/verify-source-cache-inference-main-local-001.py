"""Check every transferred main prediction and its sealed shared-cache lineage.

Raw received pixels, model execution and official AP replay are native checks.
This verifier independently checks complete transferred records and actual text
files, including their linkage to the previously closed six source caches.
"""
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'source-cache-inference-main-001'
CACHE_PREFIX = 'original-source-cache-main-001'
CACHE_NATIVE = '/mnt/d/paper6/runs/' + CACHE_PREFIX
CACHE_MIRROR = ROOT / 'data/engineering' / (CACHE_PREFIX + '-transfer')
CONDITIONS = [f'{codec}-cr{rate}' for codec in ('jpeg', 'jpeg2000') for rate in (10, 30, 50)]
CHECKPOINTS = {
    'stereo_rcnn': ('b7a07a897224f75bc30b2d4c06f39927e92933a8bcaad6e679aff605b2346c14', 670),
    'liga': ('3f36af30b686f23d0868482675177b29cc50c2b378577dd12f9092ddb0235b2e', 484),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def mapped(native):
    path = Path(native)
    if path.is_relative_to('/home/sheng/paper6'):
        return ROOT / path.relative_to('/home/sheng/paper6')
    assert path.is_relative_to('/mnt/d/paper6/runs'), native
    assert path.relative_to('/mnt/d/paper6/runs').parts[0].startswith(PREFIX + '-'), native
    return ROOT / 'data/engineering' / (PREFIX + '-native-transfer') / str(path).lstrip('/')


def check_metrics(metrics):
    assert set(metrics) == {'primary_Car_IoU_0_7_R40', 'secondary_Car_IoU_0_5', 'difficulty_order'}
    assert metrics['difficulty_order'] == ['easy', 'moderate', 'hard']
    strict = metrics['primary_Car_IoU_0_7_R40']
    expected = {f'Car_{task}/{difficulty}_R40' for task in ('image', 'bev', '3d', 'aos')
                for difficulty in ('easy', 'moderate', 'hard')}
    assert set(strict) == expected
    relaxed = metrics['secondary_Car_IoU_0_5']
    required = {f'{task}_{recall}' for task in ('bbox', 'bev', '3d') for recall in ('R11', 'R40')}
    assert required <= set(relaxed) <= required | {'aos_R11', 'aos_R40'}
    assert ('aos_R11' in relaxed) == ('aos_R40' in relaxed)
    values = list(strict.values())
    for row in relaxed.values():
        assert len(row) == 3
        values.extend(row)
    assert all(math.isfinite(value) and 0 <= value <= 100.000000001 for value in values)


def main():
    output = ROOT / 'data/provenance' / (PREFIX + '-local-verification.json')
    assert not output.exists()
    closure_path = ROOT / 'data/provenance' / (PREFIX + '-closure.json')
    closure = read(closure_path)
    assert closure['state'] == 'closed_all12_endpoints_actual_terminal_and_shared_received_inputs'
    assert closure['actual_terminal'] and closure['scope'] == 'main'
    assert closure['frames'] == 45228 and closure['shared_pairs'] == 22614
    assert len(closure['artifacts_sha256']) == 45279
    for relative, digest in closure['artifacts_sha256'].items():
        assert sha(ROOT / relative) == digest, relative
    assert {p: str(mapped(n).relative_to(ROOT)) for p, n in closure['native_to_local_artifacts'].items()} == {
        p: p for p in closure['artifacts_sha256']}
    source = closure['sources']
    roots = dict(project=ROOT, liga=ROOT / 'third_party/LIGA-Stereo',
                 mmdet=ROOT / 'third_party/mmdetection_kitti',
                 stereo_rcnn=ROOT / 'third_party/Stereo-RCNN', final_evaluation=ROOT)
    assert set(source['native']) == set(roots)
    for group, identity in source['native'].items():
        hashes = {p: sha(roots[group] / p) for p in identity['file_hashes']}
        assert hashes == identity['file_hashes']
        canonical = json.dumps(hashes, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        assert hashlib.sha256(canonical).hexdigest() == identity['sha256']
    for path, digest in source['inference'].items():
        assert sha(ROOT / path) == digest
    assert sha(ROOT / 'experiments/original-paper-scenarios/source-cache-inference-protocol-001.md') == source['protocol_sha256']
    cache_closure_path = ROOT / 'data/provenance' / (CACHE_PREFIX + '-closure.json')
    cache_closure = read(cache_closure_path)
    cache_proof = read(ROOT / 'data/provenance' / (CACHE_PREFIX + '-local-verification.json'))
    assert cache_closure['state'] == 'closed_actual_terminal_all_native_artifacts' and cache_closure['actual_terminal']
    assert cache_proof['state'] == 'passed_all22614_transferred_pair_records_and_native_audit_metadata'
    assert cache_proof['closure_sha256'] == sha(cache_closure_path)
    encoding_path = CACHE_MIRROR / (CACHE_NATIVE + '/encode.json').lstrip('/')
    assert sha(encoding_path) == cache_closure['native_files'][CACHE_NATIVE + '/encode.json']
    encoding = read(encoding_path)
    ids = encoding['frame_ids']
    assert len(ids) == len(set(ids)) == 3769
    assert all(len(frame) == 6 and frame.isdecimal() for frame in ids)
    blobs = [(separator.join(ids) + suffix).encode() for separator in ('\n', '\r\n') for suffix in ('', separator)]
    assert any(hashlib.sha256(blob).hexdigest() == '657ac4bcc1e156e5b106a4ca18e1f88e012787ea1d2b5d0adeea97fee903fa86' for blob in blobs)
    cycle_path = ROOT / 'data/runs' / (PREFIX + '-cycle.json')
    launch_path = ROOT / 'data/runs' / (PREFIX + '-launch.json')
    cycle, launch = read(cycle_path), read(launch_path)
    assert cycle['pid'] == launch['pid'] == closure['pid']
    assert cycle['sources'] == launch['sources'] == source
    assert cycle['state'] == 'finished12_native_endpoints12_audits_pending_terminal_closure'
    assert len(cycle['completed_commands']) == 24 and all(row['returncode'] == 0 for row in cycle['completed_commands'])
    assert sha(ROOT / 'data/provenance/source-cache-inference-cycle-001.py') == cycle['controller_sha256'] == launch['controller_sha256']
    expected = {condition + '-' + detector for condition in CONDITIONS for detector in CHECKPOINTS}
    assert set(cycle['endpoints']) == set(closure['endpoints']) == expected
    expected_artifacts = {str(launch_path.relative_to(ROOT)), str(cycle_path.relative_to(ROOT)),
                          str(mapped(launch['log']).relative_to(ROOT))}
    for command in cycle['completed_commands']:
        log = mapped(command['log'])
        assert sha(log) == command['log_sha256']
        expected_artifacts.add(str(log.relative_to(ROOT)))
    frames = detections = 0
    shared, ground_truth, metrics_by_endpoint = {}, {}, {}
    for condition in CONDITIONS:
        receiver_native = CACHE_NATIVE + '/received/' + condition + '/receiver.json'
        receiver_path = CACHE_MIRROR / receiver_native.lstrip('/')
        assert sha(receiver_path) == cache_closure['native_files'][receiver_native]
        received = read(receiver_path)
        assert received['state'] == 'finished_all_received_pairs' and received['frame_ids'] == ids
        for detector, (checkpoint, states) in CHECKPOINTS.items():
            label = condition + '-' + detector
            item = cycle['endpoints'][label]
            manifest, audit_path = mapped(item['manifest']), mapped(item['audit'])
            run, audit = read(manifest), read(audit_path)
            assert sha(manifest) == item['manifest_sha256'] == audit['manifest_sha256']
            assert sha(audit_path) == item['audit_sha256']
            assert run['state'] == 'finished' and audit['state'] == 'passed'
            assert run['scope'] == audit['scope'] == 'main'
            assert run['condition'] == audit['condition'] == condition and run['detector'] == audit['detector'] == detector
            assert run['sources'] == audit['sources'] == source
            assert run['protocol_sha256'] == source['protocol_sha256'] and run['seed'] == 17
            assert run['ordered_ids'] == [row['frame_id'] for row in run['frames']] == ids
            assert audit['frames'] == 3769 and set(audit['files']) == set(ids)
            assert run['cache_receiver_path'] == receiver_native and run['cache_receiver_sha256'] == sha(receiver_path)
            assert run['cache_closure_sha256'] == sha(cache_closure_path)
            assert run['inference_labels_clean_images_blocked'] and run['all_predictions_sealed_before_GT']
            assert run['started_unix'] < run['predictions_sealed_unix'] <= run['ended_unix']
            assert run['source_only_PHY_uses'] is None and run['source_only_PHY_energy'] is None
            assert not run['right_2D_measured'] and audit['all_received_float_tensors_recomputed']
            assert run['detector_checkpoint_sha256'] == checkpoint
            assert run['detector_states'] == audit['readonly_states'] == states
            assert len(run['detector_initial_state_hashes']) == states
            assert run['detector_initial_state_hashes'] == run['detector_final_state_hashes']
            assert run['peak_reserved_bytes'] + 2 * 2**30 <= run['NVIDIA_free_before_bytes']
            directory = Path(run['output_dir'])
            assert str(directory) == '/mnt/d/paper6/runs/' + PREFIX + '-' + label
            metric_path, evaluator_path = mapped(directory / 'metrics.json'), mapped(directory / 'evaluator.txt')
            assert sha(metric_path) == run['metrics_sha256'] and read(metric_path) == run['metrics'] == audit['recomputed_metrics']
            check_metrics(run['metrics'])
            metrics_by_endpoint[label] = run['metrics']
            endpoint = closure['endpoints'][label]
            assert endpoint == dict(manifest_sha256=sha(manifest), audit_sha256=sha(audit_path),
                                    frames=3769, metrics=run['metrics'], states=states)
            expected_artifacts.update(str(path.relative_to(ROOT)) for path in (manifest, audit_path, metric_path, evaluator_path))
            endpoint_detections = 0
            for row, cached in zip(run['frames'], received['records']):
                frame = row['frame_id']
                assert frame == cached['frame_id']
                prediction = mapped(directory / 'data' / (frame + '.txt'))
                expected_artifacts.add(str(prediction.relative_to(ROOT)))
                assert sha(prediction) == row['prediction_sha256'] == audit['files'][frame]['prediction_sha256']
                lines = prediction.read_text().splitlines()
                assert len(lines) == row['prediction_count']
                for line in lines:
                    fields = line.split()
                    assert len(fields) == 16 and fields[0] in (('Car',) if detector == 'stereo_rcnn' else ('Car', 'Pedestrian', 'Cyclist'))
                    numbers = [float(value) for value in fields[1:]]
                    assert all(math.isfinite(value) for value in numbers) and 0 <= numbers[-1] <= 1
                assert row['cache_path'] == cached['cache_path']
                assert row['cache_sha256'] == cached['cache_sha256'] == audit['files'][frame]['cache_sha256']
                assert row['received_arrays'] == cached['arrays']
                identity = dict(cache_sha256=row['cache_sha256'], arrays=row['received_arrays'], tensors=row['received_tensors'])
                key = (condition, frame)
                assert key not in shared or shared[key] == identity
                shared[key] = identity
                assert len(row['received_tensors']) == 2
                h, w = cached['public_hw']
                for view, tensor in zip(('left', 'right'), row['received_tensors']):
                    assert cached['arrays'][view]['dtype'] == '|u1' and cached['arrays'][view]['shape'] == [h, w, 3]
                    assert tensor['dtype'] == 'float32' and tensor['shape'] == [1, 3, h, w]
                    assert 0 <= tensor['minimum'] <= tensor['maximum'] <= 1 and tensor['out_of_range_fraction'] == 0
                truth = dict(calibration_sha256=row['calibration_sha256'], label_sha256=row['label_sha256'])
                assert frame not in ground_truth or ground_truth[frame] == truth
                ground_truth[frame] = truth
                assert all(audit['files'][frame][key] == digest for key, digest in truth.items())
                if detector == 'stereo_rcnn':
                    assert row['calls'] == dict(detector=1, image_backbone=2, dense_alignment=int(row['native_3D_counts']['initial_solutions'] > 0))
                    assert row['native_3D_counts']['dense_solutions'] == len(lines) and row['GT_placeholder_only']
                else:
                    assert row['calls'] == dict(image_backbone=2, feature_neck=2, build_cost=1, head3D=1, forbidden=0)
                frames += 1
                detections += len(lines)
                endpoint_detections += len(lines)
            assert endpoint_detections == audit['predictions']
    assert frames == 45228 and len(shared) == 22614 and len(ground_truth) == 3769
    assert set(closure['artifacts_sha256']) == expected_artifacts
    result = dict(state='passed_all45228_transferred_native_inference_records_predictions_and_metrics',
                  checked_unix=time.time(), frames=frames, predictions=detections, shared_pairs=len(shared),
                  artifacts_verified=len(expected_artifacts), sources=source, closure_sha256=sha(closure_path),
                  metrics_by_endpoint=metrics_by_endpoint, AP_measured=True, right_2D_measured=False,
                  limitation='Complete local text/record/cache-lineage checks; raw pixel conversion, CUDA model and official AP replay native only')
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(state=result['state'], frames=frames, predictions=detections)))


if __name__ == '__main__':
    main()
