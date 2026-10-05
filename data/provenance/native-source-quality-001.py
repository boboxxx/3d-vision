"""Complete source/ROI/cache-paired native quality and fresh all-record audit."""
import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path('/home/sheng/paper6')
CONDITIONS = tuple(f'{codec}-cr{rate}' for codec in ('jpeg', 'jpeg2000') for rate in (10, 30, 50))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value); return value


r = module('quality_ROI_contract', ROOT / 'experiments/original-paper-scenarios/main-ROI-code/contract.py')
m = module('quality_operators', ROOT / 'experiments/original-paper-scenarios/source-quality-code/metrics.py')


def sources():
    paths = sorted((ROOT / 'experiments/original-paper-scenarios/source-quality-code').glob('*.py'))
    assert len(paths) == 2
    paths += [r.PROTOCOL]
    return dict(quality={str(p.relative_to(ROOT)): r.sha(p) for p in paths}, ROI=r.sources(), script_sha256=r.sha(__file__))


def gate(scope):
    source = sources()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    for host in ('local', 'sheng'):
        fixture = r.read(ROOT / 'data/engineering' / ('source-quality-' + host + '-CPU-001.json'))
        assert fixture['state'] == 'passed_six_independent_RGB_quality_CPU_families' and fixture['families'] == 6
        assert fixture['sources'] == source['quality'] and fixture['maximum_SSIM_error'] < 2e-11
    roi_prefix = 'public-val-ROI-' + scope + '-001'; cache_prefix = 'original-source-cache-' + scope + '-001'
    roi_closure_path = ROOT / 'data/provenance' / (roi_prefix + '-closure.json')
    cache_closure_path = ROOT / 'data/provenance' / (cache_prefix + '-closure.json')
    roi_closure, cache_closure = r.read(roi_closure_path), r.read(cache_closure_path)
    roi_proof = r.read(ROOT / 'data/provenance' / (roi_prefix + '-local-verification.json'))
    cache_proof = r.read(ROOT / 'data/provenance' / (cache_prefix + '-local-verification.json'))
    assert roi_closure['state'] == 'closed_CPU_ROI_actual_terminal_all_native_artifacts' and roi_closure['actual_terminal']
    assert roi_proof['state'] == 'passed_all' + str(4 if scope == 'engineering' else 7538) + '_transferred_ROI_view_records'
    assert roi_proof['closure_sha256'] == r.sha(roi_closure_path) and roi_closure['sources'] == source['ROI']
    assert cache_closure['state'] == 'closed_actual_terminal_all_native_artifacts' and cache_closure['actual_terminal']
    assert cache_proof['closure_sha256'] == r.sha(cache_closure_path)
    assert cache_proof['state'] == ('passed_all_transferred_records_and_engineering_wires' if scope == 'engineering' else
                                    'passed_all22614_transferred_pair_records_and_native_audit_metadata')
    for group in cache_closure['sources'].values():
        for path, digest in group.items():
            assert r.sha(ROOT / path) == digest
    roi_run_path = ROOT / 'data/runs' / (roi_prefix + '.json'); roi_run = r.read(roi_run_path)
    assert roi_run['state'] == 'finished' and roi_run['initial_states'] == roi_run['final_states']
    records_path = Path(roi_run['records_path']); assert r.sha(records_path) == roi_run['records_sha256']
    assert r.sha(roi_run_path) == roi_closure['artifacts_sha256'][str(roi_run_path.relative_to(ROOT))]
    roi_rows = [json.loads(line) for line in records_path.read_text().splitlines()]; ids = [row['frame_id'] for row in roi_rows]
    assert ids == r.ids(scope) == roi_run['ordered_ids']
    cache = r.NATIVE / cache_prefix; encoding_path = cache / 'encode.json'
    assert r.sha(encoding_path) == cache_closure['native_files'][str(encoding_path)]
    encoding = r.read(encoding_path); assert encoding['frame_ids'] == ids
    receivers = {}
    for condition in CONDITIONS:
        path = cache / 'received' / condition / 'receiver.json'
        assert r.sha(path) == cache_closure['native_files'][str(path)]
        received = r.read(path); assert received['state'] == 'finished_all_received_pairs' and received['frame_ids'] == ids
        assert [row['frame_id'] for row in received['records']] == ids
        receivers[condition] = received
    dependencies = {str(path.relative_to(ROOT)): r.sha(path) for path in (roi_closure_path, cache_closure_path, roi_run_path)}
    return source, ids, roi_rows, encoding, receivers, dependencies


def barrier(allowed):
    allowed = {str(Path(path).resolve()) for path in allowed}
    def check(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)):
            return
        name = os.fsdecode(args[0]); mode, flags = args[1:3]
        reading = (isinstance(mode, str) and ('r' in mode or '+' in mode)) or (mode is None and flags & os.O_ACCMODE != os.O_WRONLY)
        if not reading:
            return
        assert not name.lower().endswith(('.pkl', '.pth', '.pt', '.bin'))
        assert not any(part in name for part in ('/label_2/', '/calib/', '/velodyne/'))
        if name.lower().endswith(('.png', '.npz')):
            assert str(Path(name).resolve()) in allowed
    return check


def pair(condition, source_row, roi_row, received_row):
    assert source_row['frame_id'] == roi_row['frame_id'] == received_row['frame_id']
    cache_blob = Path(received_row['cache_path']).read_bytes()
    assert hashlib.sha256(cache_blob).hexdigest() == received_row['cache_sha256']
    output = []
    with np.load(io.BytesIO(cache_blob), allow_pickle=False) as archive:
        assert set(archive.files) == {'left', 'right'}
        for name, camera, view in zip(('left', 'right'), ('image_2', 'image_3'), roi_row['views']):
            assert view['camera'] == camera
            path = r.DATA / 'training' / camera / (roi_row['frame_id'] + '.png'); blob = path.read_bytes()
            assert hashlib.sha256(blob).hexdigest() == source_row['source_images_sha256'][str(path)] == view['image_sha256']
            with Image.open(io.BytesIO(blob)) as png:
                assert png.mode == 'RGB'; target = np.asarray(png)
            actual = archive[name]
            assert actual.dtype == target.dtype == np.uint8 and actual.shape == target.shape
            assert r.describe(target) == view['native_RGB8'] == source_row['evidence'][name + '_input']
            assert r.describe(actual) == received_row['arrays'][name] == source_row['evidence'][name + '_received']
            assert list(actual.shape[:2]) == view['native_hw'] == received_row['public_hw']
            mask = r.mask(view['native_hw'], view['boxes'])
            assert r.describe(mask) == view['mask_uint8'] and int(mask.sum()) == view['union_area_pixels']
            quality = m.measure(target, actual, mask, 'RGB8')
            # Independent integer RGB8 error avoids repeating normalized floating
            # subtraction; verify every selected full/key SSE and MSE.
            difference = target.astype(np.int32) - actual.astype(np.int32)
            integer_error = difference.astype(np.int64) ** 2
            for region, selection in (('global_g', np.ones(mask.shape, dtype=bool)), ('key_k', mask.astype(bool))):
                statistics = quality[region]; count = int(selection.sum()) * 3
                expected = float(integer_error[selection].sum(dtype=np.int64)) / (255 ** 2)
                assert math.isclose(statistics['squared_error_sum'], expected, rel_tol=2e-12, abs_tol=2e-10)
                assert statistics['channel_values'] == count
            output.append(dict(frame_id=roi_row['frame_id'], condition=condition, camera=camera,
                               source_PNG_sha256=view['image_sha256'], source_RGB8=view['native_RGB8'],
                               ROI_mask=view['mask_uint8'], cache_sha256=received_row['cache_sha256'],
                               received_RGB8=received_row['arrays'][name], wire_sha256=received_row['wire_sha256'],
                               quality=quality))
    return output


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--scope', choices=('engineering', 'main'), required=True)
    parser.add_argument('--mode', choices=('measure', 'audit'), required=True); args = parser.parse_args()
    source, ids, rois, encoding, receivers, dependencies = gate(args.scope)
    prefix = 'source-quality-' + args.scope + '-001'; directory = r.NATIVE / prefix
    manifest = ROOT / 'data/runs' / (prefix + '.json'); raw = directory / 'quality.jsonl'
    audit = ROOT / 'data/provenance' / (prefix + '-audit.json')
    allowed = [r.DATA / 'training' / camera / (frame + '.png') for frame in ids for camera in ('image_2', 'image_3')]
    allowed += [row['cache_path'] for value in receivers.values() for row in value['records']]
    sys.addaudithook(barrier(allowed))
    if args.mode == 'measure':
        assert not directory.exists() and not manifest.exists(); directory.mkdir()
        if args.scope == 'main':
            proof = r.read(ROOT / 'data/provenance/source-quality-engineering-001-local-verification.json')
            assert proof['state'] == 'passed_all24_transferred_native_quality_records' and proof['sources'] == source
        run = dict(state='running', scope=args.scope, pid=os.getpid(), sources=source, dependencies=dependencies,
                   ordered_ids=ids, records_path=str(raw), started_unix=time.time(), views_done=0, conditions={},
                   GPU_used=False, GT_calibration_LiDAR_model_read=False, source_only_PHY_uses=None, source_only_PHY_energy=None)
        r.save(manifest, run)
        try:
            with raw.open('x') as stream:
                for condition in CONDITIONS:
                    values = []
                    for index, (encoded, roi, received) in enumerate(zip(encoding['conditions'][condition]['records'], rois,
                                                                        receivers[condition]['records']), 1):
                        records = pair(condition, encoded, roi, received)
                        for row in records:
                            stream.write(json.dumps(row, allow_nan=False) + '\n'); values.append(row['quality'])
                        stream.flush(); run['views_done'] += 2
                        if index % 100 == 0:
                            r.save(manifest, run); print(json.dumps(dict(condition=condition, pairs=index)), flush=True)
                    assert len(values) == 2 * len(ids)
                    run['conditions'][condition] = {region: m.aggregate(values, region) for region in ('global_g', 'key_k')}
                    r.save(manifest, run)
            assert sources() == source
            run.update(state='finished_all_six_native_quality_conditions', ended_unix=time.time(), records_sha256=r.sha(raw), read_barrier=True)
            r.save(manifest, run)
        except BaseException as error:
            run.update(state='failed', error=repr(error), ended_unix=time.time()); r.save(manifest, run); raise
        print(json.dumps(dict(state=run['state'], views=run['views_done']))); return
    assert not audit.exists(); run = r.read(manifest)
    assert run['state'] == 'finished_all_six_native_quality_conditions' and run['sources'] == source and run['dependencies'] == dependencies
    assert run['ordered_ids'] == ids and run['views_done'] == 12 * len(ids) and run['read_barrier']
    assert r.sha(raw) == run['records_sha256']
    count = 0; recomputed = {}
    with raw.open() as stream:
        for condition in CONDITIONS:
            values = []
            for encoded, roi, received in zip(encoding['conditions'][condition]['records'], rois, receivers[condition]['records']):
                # Complete fresh pixel/geometry/quality replay using the CPU-
                # independently verified frozen SSIM operator. Integer MSE is
                # independent; full SSIM reuses that same pinned operator.
                for expected in pair(condition, encoded, roi, received):
                    actual = json.loads(next(stream)); assert actual == expected
                    values.append(actual['quality']); count += 1
            recomputed[condition] = {region: m.aggregate(values, region) for region in ('global_g', 'key_k')}
        assert stream.read() == ''
    assert count == run['views_done'] and recomputed == run['conditions'] and sources() == source
    result = dict(state='passed_every_native_quality_record_fresh_pixels_integer_MSE_and_frozen_SSIM_replay', scope=args.scope,
                  checked_unix=time.time(), sources=source, dependencies=dependencies, views=count,
                  manifest_sha256=r.sha(manifest), records_sha256=r.sha(raw), conditions=recomputed,
                  independent_RGB8_integer_MSE=True,
                  limitation='Fresh complete native pixels/ROI/cache replay and independent integer MSE; full SSIM reuses the CPU literal-window-verified frozen operator. No GPU/AP/radio.')
    with audit.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(state=result['state'], views=count)))


if __name__ == '__main__':
    main()
