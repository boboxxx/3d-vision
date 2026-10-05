"""Fixed source identities, train-only sampling and patch preprocessing."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = Path('/mnt/d/paper6/data/kitti')
NATIVE = Path('/mnt/d/paper6/runs')
INTERFACE = HERE.parent / 'srcnn-compression-interface-code'
CORE = HERE.parent / 'srcnn-author-core-code'
sys.path.insert(0, str(INTERFACE))
sys.path.insert(0, str(CORE))
import interface
from core import AuthorCore, load
from model import Srcnn

MODEL = ROOT / 'third_party/SRCNN-official-001/SRCNN_v1/SRCNN/model/9-5-5(ImageNet)/x3.mat'
PROTOCOL = HERE.parent / 'srcnn-KITTI-adaptation-protocol-001.md'
TRAIN_SHA = 'b6417a1d9b18c8fdb085128e633d28ff321b7674a6d1b3841b8f43d865b281cb'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def sources():
    files = list(HERE.glob('*.py')) + list(INTERFACE.glob('*.py')) + list(CORE.glob('*.py'))
    assert len(list(HERE.glob('*.py'))) == 5
    files += [PROTOCOL, HERE.parent / 'srcnn-compression-interface-protocol-001.md',
              HERE.parent / 'srcnn-author-core-protocol-001.md', MODEL,
              MODEL.parents[2] / 'SRCNN.m']
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(files)}


def describe(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous().numpy()
    value = np.ascontiguousarray(value)
    schema = dict(dtype=value.dtype.str, shape=list(value.shape))
    return dict(**schema, sha256=hashlib.sha256(json.dumps(schema, sort_keys=True).encode() + b'\0' + value.tobytes()).hexdigest())


def states(model):
    return {key: describe(value) for key, value in model.state_dict().items()}


def initial_model():
    return Srcnn(AuthorCore(load(MODEL)))


def ids(scope):
    if scope == 'engineering':
        return ['000000', '000003']
    path = DATA / 'ImageSets/train.txt'
    assert sha(path) == TRAIN_SHA
    items = path.read_text().splitlines()
    assert len(items) == len(set(items)) == 3712
    return items


def order(frame_ids, epochs):
    views = [(frame, camera) for frame in frame_ids for camera in ('image_2', 'image_3')]
    assert len(views) % 4 == 0
    generator = np.random.Generator(np.random.PCG64(17))
    step = 0
    for epoch in range(1, epochs + 1):
        shuffled = generator.permutation(len(views))
        for offset in range(0, len(views), 4):
            step += 1
            yield epoch, step, [views[int(index)] for index in shuffled[offset:offset + 4]]


def prepare(rgb, rate):
    assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
    height, width = map(int, rgb.shape[:2])
    assert height >= 49 and width >= 49
    h, w = interface.low_hw(height, width, rate)
    low = np.asarray(Image.fromarray(rgb).resize((w, h), Image.Resampling.BICUBIC), dtype=np.uint8)
    degraded = interface.rgb_to_ycbcr(interface.interpolate(low, height, width))[:, :, 0].astype(np.float32)
    target = interface.rgb_to_ycbcr(rgb.astype(np.float64) / 255)[:, :, 0].astype(np.float32)
    return degraded, target


def patches(images, rate, generator):
    inputs, targets, records = [], [], []
    assert len(images) == 4
    for rgb in images:
        degraded, target = prepare(rgb, rate)
        height, width = degraded.shape
        coordinates = []
        for _ in range(8):
            top = int(generator.integers(0, height - 49 + 1))
            left = int(generator.integers(0, width - 49 + 1))
            coordinates.append([top, left])
            inputs.append(degraded[top:top + 49, left:left + 49])
            targets.append(target[top:top + 49, left:left + 49])
        records.append(dict(native_hw=[height, width], patch_coordinates=coordinates,
                            native_RGB8=describe(rgb)))
    return torch.from_numpy(np.stack(inputs)[:, None].copy()), torch.from_numpy(np.stack(targets)[:, None].copy()), records
