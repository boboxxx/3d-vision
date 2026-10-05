"""All transferred ROI records, native PNG audit lineage and rectangle unions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/original-paper-scenarios/main-ROI-code'))
import contract as c
from audit import union


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--scope', choices=('engineering', 'main'), required=True)
    args = parser.parse_args(); prefix = 'public-val-ROI-' + args.scope + '-001'
    output = ROOT / 'data/provenance' / (prefix + '-local-verification.json'); assert not output.exists()
    closure_path = ROOT / 'data/provenance' / (prefix + '-closure.json'); closure = c.read(closure_path)
    assert closure['state'] == 'closed_CPU_ROI_actual_terminal_all_native_artifacts' and closure['actual_terminal']
    assert closure['scope'] == args.scope and not closure['GPU_used'] and closure['sources'] == c.sources()
    frames = 2 if args.scope == 'engineering' else 3769
    assert closure['frames'] == frames and closure['views'] == 2 * frames
    assert len(closure['artifacts_sha256']) == 8
    def mapped(native):
        path = Path(native)
        if path.is_relative_to('/home/sheng/paper6'):
            return ROOT / path.relative_to('/home/sheng/paper6')
        assert path.is_relative_to('/mnt/d/paper6/runs/' + prefix)
        return ROOT / 'data/engineering' / (prefix + '-native-transfer') / str(path).lstrip('/')
    assert {p: str(mapped(n).relative_to(ROOT)) for p, n in closure['native_to_local_artifacts'].items()} == {
        p: p for p in closure['artifacts_sha256']}
    for path, digest in closure['artifacts_sha256'].items():
        assert c.sha(ROOT / path) == digest, path
    cycle_path = ROOT / 'data/runs' / (prefix + '-cycle.json'); launch_path = ROOT / 'data/runs' / (prefix + '-launch.json')
    cycle, launch = c.read(cycle_path), c.read(launch_path)
    assert cycle['pid'] == launch['pid'] == closure['pid']
    assert cycle['sources'] == launch['sources'] == closure['sources']
    assert cycle['state'] == 'finished_CPU_ROI_and_audit_pending_terminal_closure'
    assert cycle['controller_sha256'] == launch['controller_sha256'] == c.sha(ROOT / 'data/provenance/main-ROI-cycle-001.py')
    assert len(cycle['completed_commands']) == 2 and all(row['returncode'] == 0 for row in cycle['completed_commands'])
    for row in cycle['completed_commands']:
        assert c.sha(mapped(row['log'])) == row['log_sha256']
    manifest_path = ROOT / 'data/runs' / (prefix + '.json'); audit_path = ROOT / 'data/provenance' / (prefix + '-audit.json')
    run, report = c.read(manifest_path), c.read(audit_path)
    assert run['state'] == 'finished' and report['state'] == 'passed_all_native_PNG_pixels_and_independent_ROI_union_masks'
    assert run['scope'] == report['scope'] == args.scope and run['sources'] == report['sources'] == closure['sources']
    assert run['initial_states'] == run['final_states'] and report['readonly_model_states'] == len(run['initial_states'])
    assert run['model_parameters'] == 1867405 and run['device'] == 'cpu' and run['threads'] == 2 and not run['GPU_used']
    assert run['readonly_model'] and run['read_barrier'] and run['complete_native_PIL_cv2_pixel_identity']
    assert not run['GT_calibration_LiDAR_received_images_read']
    assert run['frames_done'] == report['frames'] == frames and run['views_done'] == report['views'] == 2 * frames
    assert report['manifest_sha256'] == c.sha(manifest_path)
    records_path = mapped(run['records_path']); assert c.sha(records_path) == run['records_sha256'] == report['records_sha256']
    text = records_path.read_text(); assert text.endswith('\n')
    rows = [json.loads(line) for line in text.splitlines()]; ids = [row['frame_id'] for row in rows]
    assert ids == run['ordered_ids'] and len(ids) == len(set(ids)) == frames
    if args.scope == 'engineering':
        assert ids == ['000000', '000003']
    else:
        assert all(len(frame) == 6 and frame.isdecimal() for frame in ids)
        assert any(hashlib.sha256((sep.join(ids) + ending).encode()).hexdigest() ==
                   '657ac4bcc1e156e5b106a4ca18e1f88e012787ea1d2b5d0adeea97fee903fa86'
                   for sep in ('\n', '\r\n') for ending in ('', sep))
    source_pngs, area, pixels, values, boxes, empty = {}, 0, 0, 0, 0, 0
    for row in rows:
        assert [view['camera'] for view in row['views']] == ['image_2', 'image_3']
        for view in row['views']:
            height, width = view['native_hw']; image = view['native_RGB8']
            assert height > 0 and width > 0 and image['dtype'] == '|u1' and image['shape'] == [height, width, 3]
            mask = union([height, width], view['boxes'])
            assert c.describe(mask) == view['mask_uint8'] and int(mask.sum()) == view['union_area_pixels']
            assert view['letterbox_shape'][:2] == [1, 3]
            assert all(size > 0 and size <= 640 and size % 32 == 0 for size in view['letterbox_shape'][2:])
            assert view['actual_YOLO_input']['dtype'] == '<f4' and view['actual_YOLO_input']['shape'] == view['letterbox_shape']
            path = '/mnt/d/paper6/data/kitti/training/' + view['camera'] + '/' + row['frame_id'] + '.png'
            source_pngs[path] = view['image_sha256']; area += int(mask.sum()); pixels += mask.size
            values += mask.size * 3; boxes += len(view['boxes']); empty += int(not view['boxes'])
    assert report['all_source_PNG_sha256'] == source_pngs and len(source_pngs) == 2 * frames
    assert report['union_pixels'] == area and report['full_pixels'] == pixels and report['actual_RGB_values_verified'] == values
    assert report['boxes'] == boxes and report['empty_views'] == empty and report['pooled_union_fraction'] == area / pixels
    assert closure['sources'] == c.sources()
    result = dict(state='passed_all' + str(2 * frames) + '_transferred_ROI_view_records', checked_unix=time.time(),
                  scope=args.scope, closure_sha256=c.sha(closure_path), sources=closure['sources'], artifacts=8,
                  frames=frames, views=2 * frames, union_pixels=area, full_pixels=pixels,
                  pooled_union_fraction=area / pixels, boxes=boxes, empty_views=empty,
                  readonly_model_states=report['readonly_model_states'], actual_native_RGB_values_verified=values,
                  limitation='Every saved rectangle union reconstructed locally, complete native PNG/model/audit identity verified; no local PNG or YOLO replay, quality or AP')
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({key: result[key] for key in ('state', 'views', 'empty_views', 'boxes')}))


if __name__ == '__main__':
    main()
