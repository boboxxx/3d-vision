"""Independent union/overlap/edge/invalid and actual file-read barrier fixtures."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

import contract as c
from audit import union


def rejected(function):
    try:
        function()
    except AssertionError:
        return
    raise AssertionError('Forbidden ROI input accepted')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); assert not args.output.exists(); source = c.sources()
    sets = [[], [dict(xyxy=[0, 0, 9, 7], confidence=.25, class_id=2)],
            [dict(xyxy=[0, 0, 4, 3], confidence=.9, class_id=5),
             dict(xyxy=[2, 1, 9, 7], confidence=.8, class_id=7),
             dict(xyxy=[2, 1, 9, 7], confidence=.8, class_id=7)]]
    for boxes in sets:
        literal = np.array([[any(b['xyxy'][0] <= x < b['xyxy'][2] and b['xyxy'][1] <= y < b['xyxy'][3]
                                 for b in boxes) for x in range(9)] for y in range(7)], dtype=np.uint8)
        assert np.array_equal(c.mask([7, 9], boxes), literal)
        assert np.array_equal(union([7, 9], boxes), literal)
    for box in (dict(xyxy=[0, 0, 0, 1], confidence=.9, class_id=2),
                dict(xyxy=[-1, 0, 2, 1], confidence=.9, class_id=2),
                dict(xyxy=[0, 0, 10, 1], confidence=.9, class_id=2),
                dict(xyxy=[0, 0, 2, 1], confidence=.2, class_id=2),
                dict(xyxy=[0, 0, 2, 1], confidence=.9, class_id=1)):
        rejected(lambda: c.mask([7, 9], [box])); rejected(lambda: union([7, 9], [box]))
    with tempfile.TemporaryDirectory(prefix='paper6-ROI-') as directory:
        path = Path(directory) / 'sensor.png'; path.write_bytes(b'sensor')
        control = dict(active=True); sys.addaudithook(c.guard([path], control))
        assert path.read_bytes() == b'sensor'
        for name in ('foreign.png', 'received.npz', 'weights.pt', 'label_2/a.txt', 'calib/a.txt', 'velodyne/a.bin'):
            rejected(lambda: (Path(directory) / name).read_bytes())
        control['active'] = False
    assert c.sources() == source
    result = dict(state='passed_four_sensor_ROI_CPU_families', checked_unix=time.time(), families=4, sources=source,
                  scope='Literal pixel oracle, separate prefix-sum union, invalid geometry and actual read barrier; no YOLO/KITTI/AP')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(state=result['state'], families=4)))


if __name__ == '__main__':
    main()
