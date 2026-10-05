"""Every transferred native quality row, source lineage and pooled aggregates."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/original-paper-scenarios/main-ROI-code'))
import contract as r
spec = importlib.util.spec_from_file_location('verified_quality_operators', ROOT / 'experiments/original-paper-scenarios/source-quality-code/metrics.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
CONDITIONS = tuple(f'{codec}-cr{rate}' for codec in ('jpeg', 'jpeg2000') for rate in (10, 30, 50))


def equal(left, right):
    if isinstance(left, dict):
        assert isinstance(right, dict) and set(left) == set(right)
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, float):
        assert math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=2e-12, abs_tol=2e-12)
    else:
        assert left == right


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--scope', choices=('engineering', 'main'), required=True)
    args = parser.parse_args(); prefix = 'source-quality-' + args.scope + '-001'
    closure_path = ROOT / 'data/provenance' / (prefix + '-closure.json'); closure = r.read(closure_path)
    output = ROOT / 'data/provenance' / (prefix + '-local-verification.json'); assert not output.exists()
    assert closure['state'] == 'closed_native_quality_actual_terminal_all_records' and closure['actual_terminal']
    assert closure['scope'] == args.scope and len(closure['artifacts_sha256']) == 3
    source = closure['sources']; assert source['ROI'] == r.sources()
    for path, digest in source['quality'].items():
        assert r.sha(ROOT / path) == digest
    assert source['script_sha256'] == r.sha(ROOT / 'data/provenance/native-source-quality-001.py')
    assert closure['closure_script_sha256'] == r.sha(ROOT / 'data/provenance/close-source-quality-native-001.py')
    for path, digest in closure['dependencies'].items():
        assert r.sha(ROOT / path) == digest
    def mapped(native):
        path = Path(native)
        if path.is_relative_to('/home/sheng/paper6'):
            return ROOT / path.relative_to('/home/sheng/paper6')
        assert path.is_relative_to('/mnt/d/paper6/runs/' + prefix)
        return ROOT / 'data/engineering' / (prefix + '-native-transfer') / str(path).lstrip('/')
    assert {p: str(mapped(n).relative_to(ROOT)) for p, n in closure['native_to_local_artifacts'].items()} == {
        p: p for p in closure['artifacts_sha256']}
    for path, digest in closure['artifacts_sha256'].items():
        assert r.sha(ROOT / path) == digest
    manifest_path = ROOT / 'data/runs' / (prefix + '.json'); audit_path = ROOT / 'data/provenance' / (prefix + '-audit.json')
    run, report = r.read(manifest_path), r.read(audit_path)
    assert run['state'] == 'finished_all_six_native_quality_conditions'
    assert report['state'] == 'passed_every_native_quality_record_fresh_pixels_integer_MSE_and_frozen_SSIM_replay'
    assert run['sources'] == report['sources'] == source and run['dependencies'] == report['dependencies'] == closure['dependencies']
    assert run['scope'] == report['scope'] == args.scope and report['independent_RGB8_integer_MSE']
    assert run['read_barrier'] and not run['GPU_used'] and not run['GT_calibration_LiDAR_model_read']
    assert report['manifest_sha256'] == r.sha(manifest_path)
    ids = run['ordered_ids']; frames = 2 if args.scope == 'engineering' else 3769
    assert len(ids) == len(set(ids)) == frames
    if args.scope == 'engineering':
        assert ids == ['000000', '000003']
    else:
        assert any(hashlib.sha256((sep.join(ids) + end).encode()).hexdigest() ==
                   '657ac4bcc1e156e5b106a4ca18e1f88e012787ea1d2b5d0adeea97fee903fa86'
                   for sep in ('\n', '\r\n') for end in ('', sep))
    roi_run = r.read(ROOT / 'data/runs' / ('public-val-ROI-' + args.scope + '-001.json'))
    roi_prefix = 'public-val-ROI-' + args.scope + '-001'
    roi_native = Path(roi_run['records_path'])
    assert roi_native.is_relative_to('/mnt/d/paper6/runs/' + roi_prefix)
    roi_records = ROOT / 'data/engineering' / (roi_prefix + '-native-transfer') / str(roi_native).lstrip('/')
    assert r.sha(roi_records) == roi_run['records_sha256']
    rois = [json.loads(line) for line in roi_records.read_text().splitlines()]
    assert [row['frame_id'] for row in rois] == ids
    cache_prefix = 'original-source-cache-' + args.scope + '-001'
    cache_closure = r.read(ROOT / 'data/provenance' / (cache_prefix + '-closure.json'))
    cache_mirror = ROOT / 'data/engineering' / (cache_prefix + '-transfer')
    directory = '/mnt/d/paper6/runs/' + cache_prefix
    encoding = r.read(cache_mirror / (directory + '/encode.json').lstrip('/'))
    assert encoding['frame_ids'] == ids
    raw = mapped(run['records_path']); assert r.sha(raw) == run['records_sha256'] == report['records_sha256']
    count = 0; summaries = {}
    with raw.open() as stream:
        for condition in CONDITIONS:
            receiver_path = directory + '/received/' + condition + '/receiver.json'
            path = cache_mirror / receiver_path.lstrip('/'); assert r.sha(path) == cache_closure['native_files'][receiver_path]
            receiver = r.read(path); values = []
            assert [row['frame_id'] for row in receiver['records']] == ids
            for encoded, roi, received in zip(encoding['conditions'][condition]['records'], rois, receiver['records']):
                for name, camera, view in zip(('left', 'right'), ('image_2', 'image_3'), roi['views']):
                    row = json.loads(next(stream)); count += 1
                    assert row['frame_id'] == encoded['frame_id'] == roi['frame_id'] == received['frame_id']
                    assert row['condition'] == condition and row['camera'] == camera == view['camera']
                    native_png = '/mnt/d/paper6/data/kitti/training/' + camera + '/' + row['frame_id'] + '.png'
                    assert row['source_PNG_sha256'] == view['image_sha256'] == encoded['source_images_sha256'][native_png]
                    assert row['source_RGB8'] == view['native_RGB8'] == encoded['evidence'][name + '_input']
                    assert row['ROI_mask'] == view['mask_uint8'] and row['cache_sha256'] == received['cache_sha256']
                    assert row['received_RGB8'] == received['arrays'][name] == encoded['evidence'][name + '_received']
                    assert row['wire_sha256'] == received['wire_sha256'] == encoded['evidence']['wire_sha256']
                    quality = row['quality']; assert quality['quality_policy'] == 'RGB8' and quality['native_hw'] == view['native_hw']
                    assert quality['received_out_of_range_fraction'] == 0
                    for region, centers in (('global_g', math.prod(view['native_hw'])), ('key_k', view['union_area_pixels'])):
                        statistics = quality[region]
                        assert statistics['pixel_centers'] == centers and statistics['channel_values'] == 3 * centers
                        assert statistics['undefined_empty_ROI'] == (centers == 0)
                        if centers == 0:
                            assert statistics['mse'] is None and statistics['ssim'] is None and statistics['psnr_dB'] is None
                        else:
                            equal(statistics['mse'], statistics['squared_error_sum'] / (3 * centers))
                            equal(statistics['ssim'], statistics['ssim_sum'] / (3 * centers))
                            assert 0 <= statistics['mse'] <= 1 and statistics['psnr_infinite'] == (statistics['mse'] == 0)
                            if statistics['mse'] == 0:
                                assert statistics['psnr_dB'] is None
                            else:
                                equal(statistics['psnr_dB'], -10 * math.log10(statistics['mse']))
                    values.append(quality)
            summaries[condition] = {region: m.aggregate(values, region) for region in ('global_g', 'key_k')}
        assert stream.read() == ''
    assert count == closure['views'] == run['views_done'] == report['views'] == frames * 12
    equal(summaries, closure['conditions']); equal(summaries, run['conditions']); equal(summaries, report['conditions'])
    result = dict(state='passed_all' + str(count) + '_transferred_native_quality_records', checked_unix=time.time(),
                  scope=args.scope, views=count, artifacts=3, sources=source, dependencies=closure['dependencies'],
                  closure_sha256=r.sha(closure_path), conditions=summaries,
                  verifier_sha256=r.sha(__file__), retained_development_path_failure='data/provenance/source-quality-local-path-failure-001.json',
                  limitation='All saved native pixel/ROI/cache lineage and numeric rows checked; pooled aggregates independently recomputed locally. Raw pixels/integer MSE/SSIM replay occurred natively, not locally.')
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(state=result['state'], views=count)))


if __name__ == '__main__':
    main()
