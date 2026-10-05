"""Explicit native RGB quality, reflect Gaussian SSIM and pooled statistics."""
import math

import numpy as np
from scipy.ndimage import convolve1d

POLICIES = ('RGB8', 'neural_RGB8_quality', 'ECSIC_clipped_float', 'unclipped_float_extension')
COORDINATES = np.arange(-5, 6, dtype=np.float64)
GAUSSIAN = np.exp(-(COORDINATES ** 2) / (2 * 1.5 ** 2))
GAUSSIAN /= GAUSSIAN.sum()


def view(received, policy):
    assert policy in POLICIES and isinstance(received, np.ndarray)
    assert received.ndim == 3 and received.shape[-1] == 3 and all(x > 0 for x in received.shape)
    assert np.isfinite(received).all()
    if policy == 'RGB8':
        assert received.dtype == np.uint8
        return received.astype(np.float64) / 255
    assert received.dtype in (np.dtype('<f4'), np.dtype('<f8'))
    value = received.astype(np.float64)
    if policy == 'neural_RGB8_quality':
        return np.rint(255 * np.clip(value, 0, 1)) / 255
    if policy == 'ECSIC_clipped_float':
        return np.clip(value, 0, 1)
    return value


def smooth(value):
    assert value.dtype == np.float64
    return convolve1d(convolve1d(value, GAUSSIAN, axis=0, mode='reflect'), GAUSSIAN, axis=1, mode='reflect')


def ssim_map(clean, received):
    assert clean.dtype == received.dtype == np.float64 and clean.shape == received.shape
    assert clean.ndim == 3 and clean.shape[-1] == 3 and np.isfinite(clean).all() and np.isfinite(received).all()
    mx, my = smooth(clean), smooth(received)
    vx, vy, cov = smooth(clean * clean) - mx * mx, smooth(received * received) - my * my, smooth(clean * received) - mx * my
    value = ((2 * mx * my + .01 ** 2) * (2 * cov + .03 ** 2)) / (
        (mx * mx + my * my + .01 ** 2) * (vx + vy + .03 ** 2))
    assert value.shape == clean.shape and np.isfinite(value).all()
    return value


def statistics(error, ssim, mask):
    assert error.shape == ssim.shape and mask.shape == error.shape[:2]
    centers = int(mask.sum()); count = centers * 3
    if count == 0:
        return dict(pixel_centers=0, channel_values=0, squared_error_sum=0.0, ssim_sum=0.0,
                    mse=None, psnr_dB=None, psnr_infinite=False, ssim=None, undefined_empty_ROI=True)
    squared_error_sum = float(error[mask].sum(dtype=np.float64)); ssim_sum = float(ssim[mask].sum(dtype=np.float64))
    mse = squared_error_sum / count
    return dict(pixel_centers=centers, channel_values=count, squared_error_sum=squared_error_sum, ssim_sum=ssim_sum,
                mse=mse, psnr_dB=None if mse == 0 else -10 * math.log10(mse), psnr_infinite=mse == 0,
                ssim=ssim_sum / count, undefined_empty_ROI=False)


def measure(clean_RGB8, received, ROI, policy):
    assert clean_RGB8.dtype == np.uint8 and clean_RGB8.ndim == 3 and clean_RGB8.shape[-1] == 3
    assert received.shape == clean_RGB8.shape and ROI.shape == clean_RGB8.shape[:2]
    assert ROI.dtype in (np.dtype('bool'), np.dtype('uint8')) and np.all((ROI == 0) | (ROI == 1))
    target = clean_RGB8.astype(np.float64) / 255; actual = view(received, policy)
    error = (target - actual) ** 2; ssim = ssim_map(target, actual)
    full = np.ones(ROI.shape, dtype=bool)
    return dict(global_g=statistics(error, ssim, full), key_k=statistics(error, ssim, ROI.astype(bool)),
                quality_policy=policy, native_hw=list(ROI.shape),
                received_out_of_range_fraction=float(np.mean((received < 0) | (received > (255 if policy == 'RGB8' else 1)))))


def aggregate(records, region):
    assert records and region in ('global_g', 'key_k')
    values = [row[region] for row in records]
    count = sum(row['channel_values'] for row in values); centers = sum(row['pixel_centers'] for row in values)
    error = math.fsum(row['squared_error_sum'] for row in values); ssim = math.fsum(row['ssim_sum'] for row in values)
    finite = [row['psnr_dB'] for row in values if row['psnr_dB'] is not None]
    mse = error / count if count else None
    return dict(views=len(records), defined_views=sum(not row['undefined_empty_ROI'] for row in values),
                empty_views=sum(row['undefined_empty_ROI'] for row in values), perfect_views=sum(row['psnr_infinite'] for row in values),
                pixel_centers=centers, channel_values=count, squared_error_sum=error, ssim_sum=ssim,
                pooled_mse=mse, pooled_psnr_dB=None if not count or mse == 0 else -10 * math.log10(mse),
                pooled_psnr_infinite=bool(count and mse == 0), pooled_ssim=ssim / count if count else None,
                secondary_finite_view_psnr_mean=math.fsum(finite) / len(finite) if finite else None,
                secondary_finite_view_psnr_count=len(finite))
