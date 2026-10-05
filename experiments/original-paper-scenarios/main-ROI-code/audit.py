"""Every native PNG and independently reconstructed rectangular union mask."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

import contract as c


def union(shape, boxes):
    height, width = shape
    difference = np.zeros((height + 1, width + 1), dtype=np.int32)
    for box in boxes:
        assert set(box) == {'xyxy', 'confidence', 'class_id'}
        x1, y1, x2, y2 = box['xyxy']
        assert all(type(v) is int for v in box['xyxy'])
        assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
        assert box['class_id'] in (2, 5, 7) and np.isfinite(box['confidence']) and .25 <= box['confidence'] <= 1
        difference[y1, x1] += 1; difference[y2, x1] -= 1
        difference[y1, x2] -= 1; difference[y2, x2] += 1
    return (difference.cumsum(0).cumsum(1)[:height, :width] > 0).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--scope', choices=('engineering', 'main'), required=True)
    args = parser.parse_args(); prefix = 'public-val-ROI-' + args.scope + '-001'
    manifest = c.ROOT / 'data/runs' / (prefix + '.json')
    output = c.ROOT / 'data/provenance' / (prefix + '-audit.json'); assert not output.exists()
    run = c.read(manifest); ids = c.ids(args.scope); source = c.sources()
    assert run['state'] == 'finished' and run['scope'] == args.scope and run['ordered_ids'] == ids
    assert run['sources'] == source and run['initial_states'] == run['final_states']
    assert run['readonly_model'] and run['read_barrier'] and run['complete_native_PIL_cv2_pixel_identity']
    assert run['device'] == 'cpu' and run['threads'] == 2 and not run['GPU_used']
    assert not run['GT_calibration_LiDAR_received_images_read']
    assert run['frames_done'] == len(ids) and run['views_done'] == 2 * len(ids)
    records = Path(run['records_path']); assert c.sha(records) == run['records_sha256']
    text = records.read_text(); assert text.endswith('\n')
    rows = [json.loads(line) for line in text.splitlines()]; assert [row['frame_id'] for row in rows] == ids
    files, area, pixels, total_boxes, empty, pixel_values = {}, 0, 0, 0, 0, 0
    for row in rows:
        assert [view['camera'] for view in row['views']] == ['image_2', 'image_3']
        for view in row['views']:
            path = c.DATA / 'training' / view['camera'] / (row['frame_id'] + '.png')
            assert c.sha(path) == view['image_sha256']
            with Image.open(path) as png:
                assert png.mode == 'RGB'; rgb = np.asarray(png)
            assert c.describe(rgb) == view['native_RGB8'] and list(rgb.shape[:2]) == view['native_hw']
            assert view['letterbox_shape'][:2] == [1, 3]
            assert all(x > 0 and x <= 640 and x % 32 == 0 for x in view['letterbox_shape'][2:])
            assert view['actual_YOLO_input']['dtype'] == '<f4' and view['actual_YOLO_input']['shape'] == view['letterbox_shape']
            assert len(view['boxes']) <= 300
            mask = union(view['native_hw'], view['boxes'])
            assert c.describe(mask) == view['mask_uint8'] and int(mask.sum()) == view['union_area_pixels']
            files[str(path)] = view['image_sha256']; area += int(mask.sum()); pixels += mask.size
            pixel_values += rgb.size; total_boxes += len(view['boxes']); empty += int(not view['boxes'])
    assert len(files) == 2 * len(ids) and source == c.sources()
    result = dict(state='passed_all_native_PNG_pixels_and_independent_ROI_union_masks', checked_unix=time.time(),
                  scope=args.scope, sources=source, frames=len(ids), views=2 * len(ids), actual_RGB_values_verified=pixel_values,
                  manifest_sha256=c.sha(manifest), records_sha256=c.sha(records), all_source_PNG_sha256=files,
                  union_pixels=area, full_pixels=pixels, pooled_union_fraction=area / pixels,
                  boxes=total_boxes, empty_views=empty, readonly_model_states=len(run['initial_states']),
                  limitation='Actual complete pixel and saved rectangle union identity; no fresh YOLO inference, GT, quality or AP')
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(state=result['state'], frames=len(ids), views=2 * len(ids), empty_views=empty)))


if __name__ == '__main__':
    main()
