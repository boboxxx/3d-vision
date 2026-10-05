"""Fixed public ROI identities, native rectangles and sensor-only read scope."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = Path('/mnt/d/paper6/data/kitti')
NATIVE = Path('/mnt/d/paper6/runs')
METADATA = ROOT / 'reproduction/cao2025/upstream/yolov5.json'
PROTOCOL = HERE.parent / 'main-ROI-quality-protocol-001.md'
PROTOCOL_SHA = '970e5ea8b494ad356eee4ac2bdf9014fcb2fa150185bd5d702644400963a7768'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)); temporary.replace(path)


def describe(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().contiguous().numpy()
    value = np.ascontiguousarray(value); schema = dict(dtype=value.dtype.str, shape=list(value.shape))
    return dict(**schema, sha256=hashlib.sha256(json.dumps(schema, sort_keys=True).encode() + b'\0' + value.tobytes()).hexdigest())


def mask(shape, boxes):
    height, width = shape
    assert isinstance(height, int) and isinstance(width, int) and height > 0 and width > 0
    result = np.zeros((height, width), dtype=np.uint8)
    assert len(boxes) <= 300
    for box in boxes:
        assert set(box) == {'xyxy', 'confidence', 'class_id'}
        x1, y1, x2, y2 = box['xyxy']
        assert all(isinstance(v, int) for v in box['xyxy'])
        assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
        assert box['class_id'] in (2, 5, 7) and np.isfinite(box['confidence']) and .25 <= box['confidence'] <= 1
        result[y1:y2, x1:x2] = 1
    return result


def ids(scope):
    assert scope in ('engineering', 'main')
    if scope == 'engineering':
        return ['000000', '000003']
    path = DATA / 'ImageSets/val.txt'
    assert sha(path) == '657ac4bcc1e156e5b106a4ca18e1f88e012787ea1d2b5d0adeea97fee903fa86'
    values = path.read_text().splitlines(); assert len(values) == len(set(values)) == 3769
    return values


def sources():
    assert sha(PROTOCOL) == PROTOCOL_SHA
    files = sorted(HERE.glob('*.py'))
    assert len(files) == 4
    files += [PROTOCOL, METADATA, ROOT / 'reproduction/cao2025/extract_rois.py', ROOT / 'reproduction/cao2025/audit_rois.py']
    upstream = ROOT / 'third_party/yolov5'; metadata = read(METADATA)
    assert subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() == metadata['revision']
    assert not subprocess.check_output(['git', '-C', str(upstream), 'diff', '--name-only'], text=True).strip()
    tracked = subprocess.check_output(['git', '-C', str(upstream), 'ls-files'], text=True).splitlines()
    return dict(implementation={str(path.relative_to(ROOT)): sha(path) for path in files},
                upstream={path: sha(upstream / path) for path in tracked}, revision=metadata['revision'],
                checkpoint_sha256=metadata['checkpoint_sha256'])


def guard(allowed_pngs, control):
    allowed = {str(Path(path).resolve()) for path in allowed_pngs}
    def barrier(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)) or not control['active']:
            return
        name = os.fsdecode(args[0]); mode, flags = args[1:3]
        reading = (isinstance(mode, str) and ('r' in mode or '+' in mode)) or (mode is None and flags & os.O_ACCMODE != os.O_WRONLY)
        if not reading:
            return
        assert not name.lower().endswith(('.pkl', '.pth', '.pt', '.bin', '.npz'))
        assert not any(part in name for part in ('/label_2/', '/calib/', '/velodyne/'))
        if name.lower().endswith('.png'):
            assert str(Path(name).resolve()) in allowed, 'foreign sensor image read forbidden'
    return barrier
