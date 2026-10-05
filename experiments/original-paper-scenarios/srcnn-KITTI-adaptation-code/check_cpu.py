"""Independent author-filter, central-patch, gradient and sampling checks."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import PIL
from scipy.ndimage import correlate
import torch
from torch.nn import functional as F

import common as c
from model import central_loss


def reference(data, value):
    # Interpret original MAT slices directly; don't reuse converted model kernels.
    output = []
    for image in value[:, 0]:
        first = np.stack([np.maximum(correlate(image, data['weights_conv1'][:, index].reshape(9, 9, order='F').astype(np.float32), mode='nearest')
                                   + np.float32(data['biases_conv1'][index, 0]), 0) for index in range(64)])
        second = []
        for index in range(32):
            summed = np.zeros(image.shape, dtype=np.float32)
            for channel in range(64):
                kernel = data['weights_conv2'][channel, :, index].reshape(5, 5, order='F').astype(np.float32)
                summed += correlate(first[channel], kernel, mode='nearest')
            second.append(np.maximum(summed + np.float32(data['biases_conv2'][index, 0]), 0))
        summed = np.zeros(image.shape, dtype=np.float32)
        for channel in range(32):
            kernel = data['weights_conv3'][channel].reshape(5, 5, order='F').astype(np.float32)
            summed += correlate(second[channel], kernel, mode='nearest')
        output.append(summed + np.float32(data['biases_conv3'][0, 0]))
    return np.stack(output)[:, None]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); assert not args.output.exists()
    torch.set_num_threads(2)
    sources = c.sources()
    def guard(event, values):
        if event != 'open' or not isinstance(values[0], (str, bytes)):
            return
        name = os.fsdecode(values[0])
        assert not name.lower().endswith(('.png', '.pkl', '.npz', '.pth', '.pt', '.bin'))
        assert '/calib/' not in name and '/label_2/' not in name and '/velodyne/' not in name
        if name.endswith('.mat'):
            assert Path(name).resolve() == c.MODEL.resolve()
    sys.addaudithook(guard)
    model = c.initial_model(); initial = c.states(model)
    generator = np.random.Generator(np.random.PCG64(20261004))
    value = generator.uniform(16 / 255, 235 / 255, (2, 1, 16, 20)).astype(np.float32)
    with torch.no_grad():
        actual = model(torch.from_numpy(value)).numpy()
    expected = reference(c.load(c.MODEL), value)
    error = float(np.max(np.abs(actual - expected)))
    assert error < 2e-5 and np.isfinite(actual).all()
    # Patch center is unaffected by artificial patch boundaries.
    canvas = torch.from_numpy(generator.uniform(0, 1, (1, 1, 65, 73)).astype(np.float32))
    with torch.no_grad():
        complete = model(canvas)
        patch = model(canvas[:, :, 9:58, 13:62])
    assert torch.max(torch.abs(patch[:, :, 8:-8, 8:-8] - complete[:, :, 17:50, 21:54])).item() < 2e-5
    # Separately compute three valid convolutions, no replicate padding.
    direct = canvas[:, :, 9:58, 13:62]
    with torch.no_grad():
        for index in range(1, 4):
            direct = F.conv2d(direct, getattr(model, 'weight' + str(index)), getattr(model, 'bias' + str(index)))
            if index < 3:
                direct = F.relu(direct)
    assert list(direct.shape) == [1, 1, 33, 33]
    assert torch.max(torch.abs(direct - patch[:, :, 8:-8, 8:-8])).item() < 2e-5
    rows, columns = np.indices((65, 73))
    rgb = np.stack([(rows * 7 + columns * 3) % 256, (rows * 19) % 256, (columns * 11) % 256], axis=-1).astype(np.uint8)
    images = [rgb, np.roll(rgb, 3, axis=1), rgb[::-1].copy(), rgb[:, ::-1].copy()]
    batches = []
    for rate in (10, 30, 50):
        inputs, targets, meta = c.patches(images, rate, np.random.Generator(np.random.PCG64(1719)))
        assert list(inputs.shape) == list(targets.shape) == [32, 1, 49, 49]
        assert inputs.dtype == targets.dtype == torch.float32
        # Verify preprocessing against actual serialized received payload.
        wire, _ = c.interface.encode(rgb, rgb, rate)
        low, _ = c.interface.unpack(wire)
        recovered = c.interface.rgb_to_ycbcr(c.interface.interpolate(low[0], 65, 73))[:, :, 0].astype(np.float32)
        for patch_index, (top, left) in enumerate(meta[0]['patch_coordinates']):
            assert np.array_equal(inputs[patch_index, 0].numpy(), recovered[top:top + 49, left:left + 49])
        batches.append((inputs, targets, meta))
    assert torch.equal(batches[0][1], batches[1][1]) and torch.equal(batches[1][1], batches[2][1])
    assert batches[0][2] == batches[1][2] == batches[2][2]
    inputs, target, _ = batches[0]
    loss = central_loss(model(inputs), target)
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    # Analytic final-bias derivative independently checks retained-pixel loss scale.
    with torch.no_grad():
        residual = model(inputs)[:, :, 8:-8, 8:-8] - target[:, :, 8:-8, 8:-8]
    derivative = 2 * residual.double().mean().item()
    assert abs(model.bias3.grad.item() - derivative) < 1e-6
    assert c.states(model) == initial
    small_ids = ['000000', '000003']
    first = list(c.order(small_ids, 6)); second = list(c.order(small_ids, 6))
    assert first == second and len(first) == 6 and all(len(set(step[2])) == 4 for step in first)
    full_ids = [f'{i:06d}' for i in range(3712)]
    full = list(c.order(full_ids, 20)); assert len(full) == 37120
    for epoch in range(1, 21):
        rows = [pair for e, _, batch in full if e == epoch for pair in batch]
        assert len(rows) == len(set(rows)) == 7424
    assert c.sources() == sources
    result = dict(state='passed_four_SRCNN_training_CPU_families', checked_unix=time.time(), families=4,
                  sources=sources, author_initial_float32_states=initial, reference_max_abs_error=error,
                  gradients=6, formal_updates_per_rate=37120, formal_view_visits_per_rate=148480,
                  environment=dict(Python=platform.python_version(), NumPy=np.__version__, Pillow=PIL.__version__, Torch=torch.__version__),
                  no_KITTI_training_GPU_or_AP=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(state=result['state'], reference_max_abs_error=error, gradients=6)))


if __name__ == '__main__':
    main()
