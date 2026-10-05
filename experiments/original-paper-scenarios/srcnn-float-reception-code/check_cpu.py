"""Independent trained-kernel/color/framing/float-cache reception checks."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import struct
import sys
import tempfile
import time
import zlib

import numpy as np
import PIL
from PIL import Image
from scipy.ndimage import correlate
import torch

import receiver as r


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def reference(value, model):
    states = {key: x.detach().cpu().numpy() for key, x in model.state_dict().items()}
    current = value[None]
    for layer in range(1, 4):
        weight, bias = states['weight' + str(layer)], states['bias' + str(layer)]
        output = []
        for index in range(weight.shape[0]):
            result = np.zeros(value.shape, dtype=np.float32)
            for channel in range(weight.shape[1]):
                result += correlate(current[channel], weight[index, channel], mode='nearest')
            result += bias[index]
            output.append(np.maximum(result, 0) if layer < 3 else result)
        current = np.stack(output)
    return current[0]


def independent_colors(wire, model):
    # Parse literal header and use a separate Pillow/matrix implementation.
    fields = struct.unpack('>4sBBBBIIHHIII', wire[:32])
    magic, version, algorithm, rate, reserved, height, width, h, w, nl, nr, crc = fields
    assert (magic, version, algorithm, reserved) == (b'P6SR', 1, 1, 0)
    assert rate in (10, 30, 50) and nl == nr == h * w * 3
    assert len(wire) == 32 + nl + nr and zlib.crc32(wire[32:]) == crc
    matrix = np.array([[65.481, 128.553, 24.966], [-37.797, -74.203, 112.0],
                       [112.0, -93.786, -18.214]], dtype=np.float64)
    offset = np.array([16.0, 128.0, 128.0], dtype=np.float64)
    outputs, model_errors = [], []
    for blob in (wire[32:32 + nl], wire[32 + nl:]):
        low = np.frombuffer(blob, dtype=np.uint8).reshape(h, w, 3)
        channels = []
        for channel in range(3):
            values = low[:, :, channel].astype(np.float32) / np.float32(255)
            channels.append(np.asarray(Image.fromarray(values).resize((width, height), Image.Resampling.BICUBIC),
                                       dtype=np.float64))
        colors = (np.stack(channels, axis=-1) @ matrix.T + offset) / 255
        input_y = colors[:, :, 0].astype(np.float32)
        y = reference(input_y, model)
        with torch.no_grad():
            native_y = model(torch.from_numpy(input_y[None, None].copy()))[0, 0].numpy()
        model_errors.append(float(np.max(np.abs(native_y - y))))
        colors[:, :, 0] = y.astype(np.float64)
        outputs.append(((colors * 255 - offset) @ np.linalg.inv(matrix).T).astype('<f4'))
    return outputs, model_errors


def rejected(function):
    try:
        function()
    except (AssertionError, ValueError):
        return
    raise AssertionError('Invalid received input was accepted')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); assert not args.output.exists()
    assert sha(r.PROTOCOL) == r.PROTOCOL_SHA
    root = r.ROOT; prefix = 'srcnn-kitti-engineering-seed17-002'
    closure_path = root / 'data/provenance' / (prefix + '-closure.json')
    closure = read(closure_path); local = read(root / 'data/provenance' / (prefix + '-local-verification.json'))
    assert closure['actual_terminal'] and closure['state'] == 'closed_actual_terminal_all_three_rates_and_paired_training'
    assert local['state'] == 'passed_all18_transferred_SRCNN_training_records_and_terminal_artifacts'
    assert local['closure_sha256'] == sha(closure_path)
    previous = closure['sources']
    assert all(sha(root / path) == digest for path, digest in previous.items())
    sources = {str(p.relative_to(root)): sha(p) for p in sorted(r.HERE.glob('*.py'))}
    assert len(sources) == 2
    torch.set_num_threads(2)
    rows, columns = np.indices((32, 48))
    left = np.stack([(rows * 7 + columns * 3) % 256, (rows * 19 + columns * 2) % 256,
                     (columns * 11 + rows * 5) % 256], axis=-1).astype(np.uint8)
    right = np.roll(left, 3, axis=1).copy()
    records, models = [], []
    for rate in (10, 30, 50):
        run = read(root / 'data/runs' / (prefix + '-cr' + str(rate) + '.json'))
        checkpoint = Path(run['checkpoint'])
        if not checkpoint.exists():
            checkpoint = root / 'data/engineering' / (prefix + '-native-transfer') / str(checkpoint).lstrip('/')
        blob = checkpoint.read_bytes()
        model = r.model_from_checkpoint(blob, run, previous, 'engineering', rate)
        models.append(model)
        rejected(lambda: r.model_from_checkpoint(blob, run, previous, 'formal', rate))
        rejected(lambda: r.model_from_checkpoint(blob, run, previous, 'engineering', 30 if rate == 10 else 10))
        rejected(lambda: r.model_from_checkpoint(blob + b'x', run, previous, 'engineering', rate))
        wire, sent = r.interface.encode(left, right, rate)
        initial = r.states(model)
        with torch.no_grad():
            actual, metadata = r.receive(wire, model)
        expected, model_errors = independent_colors(wire, model)
        assert max(model_errors) < 2e-5
        errors = [float(np.max(np.abs(a - e))) for a, e in zip(actual, expected)]
        assert max(errors) < 3e-5
        # Different clean source after encoding cannot alter byte-only reception.
        left_before = left.copy(); left[:] = 255 - left
        with torch.no_grad():
            again, repeated = r.receive(wire, model)
        left[:] = left_before
        assert all(np.array_equal(a, b) for a, b in zip(actual, again)) and metadata == repeated
        assert initial == r.states(model) == run['final_state']
        assert metadata['native_hw'] == [32, 48] and metadata['source_bytes'] == sent['source_bytes'] == len(wire)
        assert len(wire) == 32 + 2 * np.prod(metadata['low_hw']) * 3
        for invalid in (wire[:31], wire + b'x', bytes([wire[0] ^ 1]) + wire[1:],
                        wire[:-1] + bytes([wire[-1] ^ 1])):
            with torch.no_grad():
                rejected(lambda: r.receive(invalid, model))
        packed = r.pack_pair(actual)
        row = dict(cache_sha256=r.sha_bytes(packed), arrays=metadata['arrays'], native_hw=[32, 48])
        tensors = r.tensor_pair(packed, row)
        assert all(torch.equal(tensor, torch.from_numpy(a.transpose(2, 0, 1).copy())[None])
                   for tensor, a in zip(tensors, actual))
        rejected(lambda: r.tensor_pair(packed + b'x', row))
        records.append(dict(rate=rate, checkpoint_sha256=run['checkpoint_sha256'],
                            independent_model_max_abs_errors=model_errors,
                            independent_RGB_max_abs_errors=errors, source_bytes=len(wire),
                            source_metadata=sent, received=metadata, tensors=[r.describe(t) for t in tensors]))
    # Guaranteed overshoot verifies storage and adapter without relying on model behavior.
    over = [np.array([[[-.25, 1.25, .3], [2., -2., .5]]], dtype='<f4') for _ in range(2)]
    packed = r.pack_pair(over)
    row = dict(cache_sha256=r.sha_bytes(packed), arrays={k: r.describe(a) for k, a in zip(('left', 'right'), over)},
               native_hw=[1, 2])
    tensors = r.tensor_pair(packed, row)
    assert all(t.min() == -2 and t.max() == 2 and torch.equal(t, torch.from_numpy(over[0].transpose(2, 0, 1).copy())[None]) for t in tensors)
    # Install a real file audit hook and exercise received-input and GT barriers.
    with tempfile.TemporaryDirectory(prefix='paper6-srcnn-float-') as directory:
        allowed = Path(directory) / 'received.npz'; allowed.write_bytes(packed)
        control = dict(active=True)
        sys.addaudithook(r.guard([allowed], control))
        assert allowed.read_bytes() == packed
        for name in ('clean.png', 'foreign.npz', 'foreign.p6sr', 'weights.pth', 'label_2/000000.txt',
                     'calib/000000.txt', 'velodyne/000000.bin'):
            rejected(lambda: (Path(directory) / name).read_bytes())
        control['active'] = False
    assert all(sha(root / path) == digest for path, digest in previous.items())
    assert {str(p.relative_to(root)): sha(p) for p in sorted(r.HERE.glob('*.py'))} == sources
    result = dict(state='passed_six_trained_SRCNN_float_reception_CPU_families', checked_unix=time.time(),
                  families=6, views=6, pixels=27648, sources=sources, protocol_sha256=r.PROTOCOL_SHA,
                  training_sources=previous, engineering_closure_sha256=sha(closure_path), records=records,
                  readonly_states_each=6, maximum_RGB_error=max(max(x['independent_RGB_max_abs_errors']) for x in records),
                  maximum_model_error=max(max(x['independent_model_max_abs_errors']) for x in records),
                  environment=dict(Python=platform.python_version(), Torch=torch.__version__,
                                   NumPy=np.__version__, Pillow=PIL.__version__),
                  scope='Actual engineering checkpoints, synthetic bytes and float interface; no native KITTI reception or AP')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(state=result['state'], views=6, maximum_RGB_error=result['maximum_RGB_error'])))


if __name__ == '__main__':
    main()
