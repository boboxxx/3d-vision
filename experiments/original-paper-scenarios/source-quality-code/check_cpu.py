"""Literal reflect-window population SSIM, conversion and aggregation fixtures."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import scipy

import metrics as m

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROTOCOL = HERE.parent / 'main-ROI-quality-protocol-001.md'
PROTOCOL_SHA = '970e5ea8b494ad356eee4ac2bdf9014fcb2fa150185bd5d702644400963a7768'


def sources():
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha(PROTOCOL) == PROTOCOL_SHA
    paths = sorted(HERE.glob('*.py')); assert len(paths) == 2
    return {str(path.relative_to(ROOT)): sha(path) for path in paths + [PROTOCOL]}


def literal_ssim(left, right):
    # SciPy reflect is half-sample symmetry; independently pad with NumPy
    # symmetric and enumerate every full11x11 population-weighted window.
    weights = np.array([math.exp(-(i * i) / (2 * 1.5 ** 2)) for i in range(-5, 6)], dtype=np.float64)
    weights /= weights.sum(); window = weights[:, None] * weights[None, :]
    lpad = np.pad(left, ((5, 5), (5, 5), (0, 0)), mode='symmetric')
    rpad = np.pad(right, ((5, 5), (5, 5), (0, 0)), mode='symmetric')
    output = np.empty(left.shape, dtype=np.float64)
    for row in range(left.shape[0]):
        for column in range(left.shape[1]):
            for channel in range(3):
                x = lpad[row:row + 11, column:column + 11, channel]
                y = rpad[row:row + 11, column:column + 11, channel]
                mx, my = (x * window).sum(), (y * window).sum()
                vx, vy = (((x - mx) ** 2) * window).sum(), (((y - my) ** 2) * window).sum()
                cov = ((x - mx) * (y - my) * window).sum()
                output[row, column, channel] = ((2 * mx * my + .0001) * (2 * cov + .0009)) / (
                    (mx * mx + my * my + .0001) * (vx + vy + .0009))
    return output


def rejected(function):
    try:
        function()
    except AssertionError:
        return
    raise AssertionError('Invalid quality input accepted')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); assert not args.output.exists(); source = sources()
    generator = np.random.Generator(np.random.PCG64(20261005))
    x = generator.uniform(0, 1, (9, 13, 3)).astype(np.float64)
    y = generator.uniform(0, 1, (9, 13, 3)).astype(np.float64)
    fixtures = [(x, x), (np.full_like(x, .2), np.full_like(x, .7)), (x, y)]
    impulse = np.zeros_like(x); impulse[0, 0] = [1, .5, .25]; fixtures.append((np.zeros_like(x), impulse))
    errors = []
    for left, right in fixtures:
        value = m.ssim_map(left, right); expected = literal_ssim(left, right)
        errors.append(float(np.max(np.abs(value - expected))))
        assert errors[-1] < 2e-11
    assert np.max(np.abs(m.ssim_map(x, x) - 1)) < 2e-12
    constant = (2 * .2 * .7 + .0001) / (.2 ** 2 + .7 ** 2 + .0001)
    assert np.max(np.abs(m.ssim_map(fixtures[1][0], fixtures[1][1]) - constant)) < 2e-11
    clean = np.zeros((9, 13, 3), dtype=np.uint8); mask = np.zeros((9, 13), dtype=np.uint8)
    # Overlapping boxes give a union, never repeated samples.
    mask[1:5, 2:7] = 1; mask[3:8, 4:10] = 1
    first = m.measure(clean, np.full(clean.shape, .1, dtype=np.float64), mask, 'unclipped_float_extension')
    second = m.measure(clean, np.full(clean.shape, .4, dtype=np.float64), mask, 'unclipped_float_extension')
    assert first['key_k']['pixel_centers'] == 44 and first['key_k']['channel_values'] == 132
    assert abs(first['key_k']['mse'] - .01) < 1e-15 and abs(first['key_k']['psnr_dB'] - 20) < 1e-12
    pooled = m.aggregate([first, second], 'global_g')
    assert abs(pooled['pooled_mse'] - .085) < 1e-15
    assert abs(pooled['pooled_psnr_dB'] + 10 * math.log10(.085)) < 1e-12
    assert abs(pooled['pooled_psnr_dB'] - pooled['secondary_finite_view_psnr_mean']) > 1
    empty = m.measure(clean, clean, np.zeros_like(mask), 'RGB8')
    assert empty['global_g']['psnr_infinite'] and empty['global_g']['psnr_dB'] is None
    assert empty['global_g']['mse'] == 0 and empty['key_k']['undefined_empty_ROI']
    assert empty['key_k']['mse'] is None and empty['key_k']['ssim'] is None
    all_empty = m.aggregate([empty, empty], 'key_k')
    assert all_empty['empty_views'] == 2 and all_empty['pooled_mse'] is None and all_empty['pooled_ssim'] is None
    mixed = m.aggregate([first, empty], 'global_g')
    assert mixed['perfect_views'] == 1 and mixed['secondary_finite_view_psnr_count'] == 1
    ties = np.array([.5, 1.5, 2.5, 127.5, 254.5, 255.5], dtype=np.float64) / 255
    float_rgb = np.repeat(ties.reshape(2, 3, 1), 3, axis=2)
    q = m.view(float_rgb, 'neural_RGB8_quality')
    assert np.array_equal(q[:, :, 0] * 255, np.array([[0, 2, 2], [128, 254, 255]], dtype=np.float64))
    overshoot = np.array([[[-.25, 1.25, .5]]], dtype=np.float64)
    assert np.array_equal(m.view(overshoot, 'ECSIC_clipped_float'), np.array([[[0, 1, .5]]]))
    assert np.array_equal(m.view(overshoot, 'unclipped_float_extension'), overshoot)
    assert not np.array_equal(m.view(overshoot, 'neural_RGB8_quality'), m.view(overshoot, 'ECSIC_clipped_float'))
    rejected(lambda: m.measure(clean, clean[:8], mask, 'RGB8'))
    rejected(lambda: m.measure(clean, clean, mask + 2, 'RGB8'))
    rejected(lambda: m.view(clean.astype(np.int32), 'RGB8'))
    rejected(lambda: m.view(np.full(clean.shape, np.nan), 'unclipped_float_extension'))
    assert sources() == source
    result = dict(state='passed_six_independent_RGB_quality_CPU_families', checked_unix=time.time(), families=6,
                  sources=source, SSIM_literal_max_abs_errors=errors, maximum_SSIM_error=max(errors),
                  SSIM_values_each_fixture=int(x.size), pooled_fixture=pooled,
                  empty_and_perfect_JSON_policy_verified=True, quantization_half_ties_and_overshoot_verified=True,
                  environment=dict(Python=platform.python_version(), NumPy=np.__version__, SciPy=scipy.__version__),
                  scope='Independent synthetic quality formulas and aggregation only; no KITTI/ROI received-cache quality or AP')
    json.dumps(result, allow_nan=False)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(state=result['state'], maximum_SSIM_error=max(errors))))


if __name__ == '__main__':
    main()
