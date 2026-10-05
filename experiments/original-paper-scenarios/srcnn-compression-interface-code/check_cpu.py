"""Independent synthetic byte/color/native-geometry checks, no KITTI or AP."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import sys
import time
import zlib

import numpy as np
import PIL
from PIL import Image
import torch

import interface as codec

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'experiments/original-paper-scenarios/srcnn-author-core-code'))
from core import AuthorCore, fingerprint, load


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def must_reject(callback):
    try:
        callback()
    except (AssertionError, ValueError, struct.error):
        return
    raise RuntimeError('malformed input was accepted')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(2)
    here = Path(__file__).resolve().parent
    protocol = here.parent / 'srcnn-compression-interface-protocol-001.md'
    files = list(here.glob('*.py')) + [protocol]
    files += list((here.parent / 'srcnn-author-core-code').glob('*.py'))
    author = ROOT / 'third_party/SRCNN-official-001/SRCNN_v1/SRCNN'
    model_path = author / 'model/9-5-5(ImageNet)/x3.mat'
    files += [author / 'SRCNN.m', model_path]
    sources = {str(path.relative_to(ROOT)): sha(path) for path in files}
    assert len(list(here.glob('*.py'))) == 2
    def guard(event, values):
        if event != 'open' or not isinstance(values[0], (str, bytes)):
            return
        path = os.fsdecode(values[0])
        assert not path.lower().endswith(('.png', '.pkl', '.npz', '.pth', '.pt', '.bin'))
        assert '/label_2/' not in path and '/calib/' not in path and '/velodyne/' not in path
        if path.lower().endswith('.mat'):
            assert Path(path).resolve() == model_path.resolve()
    sys.addaudithook(guard)
    model = AuthorCore(load(model_path)).eval()
    before = {key: fingerprint(value.numpy()) for key, value in model.state_dict().items()}
    assert len(before) == 6 and not list(model.parameters())
    # Independent color reference values for black, white and RGB primaries.
    colors = np.array([[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    expected = np.array([[16, 128, 128], [235, 128, 128],
                         [81.481, 90.203, 240], [144.553, 53.797, 34.214],
                         [40.966, 240, 109.786]], dtype=np.float64) / 255
    assert np.max(np.abs(codec.rgb_to_ycbcr(colors) - expected)) < 1e-14
    assert np.max(np.abs(codec.ycbcr_to_rgb(expected) - colors)) < 1e-14
    rows, columns = np.indices((32, 48))
    left = np.stack([(rows * 7 + columns * 3) % 256, (rows * 19) % 256,
                     (columns * 11) % 256], axis=-1).astype(np.uint8)
    right = np.roll(left, -3, axis=1).copy()
    records = []
    for rate, expected_low in ((10, (10, 15)), (30, (5, 8)), (50, (4, 6))):
        wire, encoded = codec.encode(left, right, rate)
        # Independent format parser and direct Pillow payload construction.
        fields = struct.unpack_from('>4sBBBBIIHHIII', wire)
        assert fields[:5] == (b'P6SR', 1, 1, rate, 0)
        assert fields[5:7] == (32, 48) and fields[7:9] == expected_low
        h, w = expected_low
        assert fields[9:11] == (h * w * 3, h * w * 3)
        assert fields[11] == zlib.crc32(wire[32:]) and len(wire) == 32 + h * w * 6
        for offset, image in ((32, left), (32 + h * w * 3, right)):
            expected_payload = Image.fromarray(image).resize((w, h), Image.Resampling.BICUBIC).tobytes()
            assert wire[offset:offset + h * w * 3] == expected_payload
        decoded, meta = codec.receive(wire, model)
        replay, _ = codec.receive(wire, model)
        assert all(np.array_equal(a, b) for a, b in zip(decoded, replay))
        assert encoded['source_bytes'] == meta['source_bytes'] == len(wire)
        # Identity Y core leaves exactly bicubic RGB, with no hidden crop/clip.
        identity, _ = codec.receive(wire, lambda value: value)
        low_images, _ = codec.unpack(wire)
        for actual, low in zip(identity, low_images):
            expected_rgb = np.empty((32, 48, 3), dtype=np.float64)
            for channel in range(3):
                plane = low[:, :, channel].astype(np.float32) / np.float32(255)
                expected_rgb[:, :, channel] = np.asarray(Image.fromarray(plane).resize((48, 32), Image.Resampling.BICUBIC))
            assert np.max(np.abs(actual - expected_rgb)) < 1e-14
        # Received-byte causality, changing one left pixel and rechecking CRC.
        altered = bytearray(wire)
        altered[32] ^= 127
        struct.pack_into('>I', altered, 28, zlib.crc32(altered[32:]))
        changed, _ = codec.receive(bytes(altered), model)
        assert not np.array_equal(changed[0], decoded[0]) and np.array_equal(changed[1], decoded[1])
        for blob in (wire[:31], wire[:-1], wire + b'\0', bytes(bytearray(wire[:32]) + bytes([wire[32] ^ 1]) + wire[33:])):
            must_reject(lambda blob=blob: codec.unpack(blob))
        for index, value in ((0, ord('X')), (4, 2), (5, 0), (6, 9), (7, 1), (19, 0)):
            bad = bytearray(wire); bad[index] = value
            must_reject(lambda bad=bad: codec.unpack(bytes(bad)))
        assert {key: fingerprint(value.numpy()) for key, value in model.state_dict().items()} == before
        records.append(dict(rate=rate, encoded=encoded, received=meta,
                            wire_sha256=hashlib.sha256(wire).hexdigest(),
                            output_arrays=[fingerprint(value) for value in decoded]))
    for shape, rate in (((1, 1), 10), ((32, 48), 9), ((0, 48), 10)):
        must_reject(lambda shape=shape, rate=rate: codec.low_hw(*shape, rate))
    must_reject(lambda: codec.encode(left, right[:, :-1], 10))
    must_reject(lambda: codec.encode(left.astype(np.float32), right, 10))
    assert all(sha(ROOT / path) == digest for path, digest in sources.items())
    result = dict(state='passed_four_synthetic_interface_check_families', checked_unix=time.time(),
                  families=4, rates=3, native_output_views=6, readonly_states=6, sources=sources,
                  environment=dict(Python=platform.python_version(), NumPy=np.__version__, Pillow=PIL.__version__, Torch=torch.__version__),
                  records=records, author_state_hashes=before, no_KITTI_GT_calibration_training_quality_or_AP=True,
                  limitation='Declared byte/color/geometry engineering only; original SRCNN compression adaptation and KITTI training unresolved')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(state=result['state'], rates=3, readonly_states=6)))


if __name__ == '__main__':
    main()
