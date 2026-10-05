"""Declared dimension-based RGB8 source and native-geometry SRCNN receiver."""
import math
import struct
import zlib

import numpy as np
from PIL import Image
import torch

HEADER = struct.Struct('>4sBBBBIIHHIII')
RATES = (10, 30, 50)
A = np.array([[65.481, 128.553, 24.966], [-37.797, -74.203, 112.0],
              [112.0, -93.786, -18.214]], dtype=np.float64)
OFFSET = np.array([16.0, 128.0, 128.0], dtype=np.float64)


def low_hw(height, width, rate):
    assert rate in RATES and isinstance(height, int) and isinstance(width, int)
    assert height > 0 and width > 0
    h, w = math.floor(height / math.sqrt(rate)), math.floor(width / math.sqrt(rate))
    assert 0 < h <= 65535 and 0 < w <= 65535
    return h, w


def encode(left, right, rate):
    assert all(isinstance(a, np.ndarray) and a.dtype == np.uint8 and a.ndim == 3 and a.shape[2] == 3
               for a in (left, right))
    assert left.shape == right.shape
    height, width = map(int, left.shape[:2])
    h, w = low_hw(height, width, rate)
    reduced = [np.asarray(Image.fromarray(a).resize((w, h), Image.Resampling.BICUBIC), dtype=np.uint8)
               for a in (left, right)]
    blobs = [a.tobytes(order='C') for a in reduced]
    payload = b''.join(blobs)
    header = HEADER.pack(b'P6SR', 1, 1, rate, 0, height, width, h, w,
                         len(blobs[0]), len(blobs[1]), zlib.crc32(payload))
    wire = header + payload
    return wire, dict(nominal_compression=rate, native_hw=[height, width], low_hw=[h, w],
                      payload_bytes=list(map(len, blobs)), framing_bytes=HEADER.size,
                      source_bytes=len(wire), source_bits=8 * len(wire),
                      raw_RGB8_bits=8 * (left.size + right.size),
                      actual_raw_to_wire_ratio=(left.size + right.size) / len(wire),
                      source_only_PHY_uses=None, source_only_PHY_energy=None)


def unpack(wire):
    assert isinstance(wire, bytes) and len(wire) >= HEADER.size
    magic, version, algorithm, rate, reserved, height, width, h, w, nl, nr, crc = HEADER.unpack_from(wire)
    assert (magic, version, algorithm, reserved) == (b'P6SR', 1, 1, 0)
    assert (h, w) == low_hw(height, width, rate)
    assert nl == nr == h * w * 3 and len(wire) == HEADER.size + nl + nr
    assert zlib.crc32(wire[HEADER.size:]) == crc
    payload = wire[HEADER.size:]
    images = [np.frombuffer(blob, dtype=np.uint8).reshape(h, w, 3).copy()
              for blob in (payload[:nl], payload[nl:])]
    return images, dict(nominal_compression=rate, native_hw=[height, width], low_hw=[h, w],
                        source_bytes=len(wire), framing_bytes=HEADER.size)


def rgb_to_ycbcr(rgb):
    assert rgb.shape[-1] == 3 and np.isfinite(rgb).all()
    return (np.asarray(rgb, dtype=np.float64) @ A.T + OFFSET) / 255


def ycbcr_to_rgb(ycbcr):
    assert ycbcr.shape[-1] == 3 and np.isfinite(ycbcr).all()
    return (np.asarray(ycbcr, dtype=np.float64) * 255 - OFFSET) @ np.linalg.inv(A).T


def interpolate(low, height, width):
    channels = []
    for index in range(3):
        channel = np.asarray(low[:, :, index], dtype=np.float32) / np.float32(255)
        channels.append(np.asarray(Image.fromarray(channel).resize((width, height), Image.Resampling.BICUBIC),
                                   dtype=np.float64))
    return np.stack(channels, axis=-1)


def receive(wire, core):
    low_views, header = unpack(wire)
    height, width = header['native_hw']
    outputs = []
    with torch.no_grad():
        for low in low_views:
            colors = rgb_to_ycbcr(interpolate(low, height, width))
            value = torch.from_numpy(colors[:, :, 0].copy()).unsqueeze(0).unsqueeze(0)
            luminance = core(value)
            assert luminance.dtype == torch.float64 and luminance.device.type == 'cpu'
            assert list(luminance.shape) == [1, 1, height, width] and torch.isfinite(luminance).all()
            colors[:, :, 0] = luminance[0, 0].numpy()
            rgb = ycbcr_to_rgb(colors)
            assert rgb.shape == (height, width, 3) and rgb.dtype == np.float64 and np.isfinite(rgb).all()
            outputs.append(rgb)
    return outputs, dict(**header, received_header_only_geometry=True, output_unclipped=True,
                          ranges=[dict(minimum=float(a.min()), maximum=float(a.max()),
                                       out_of_range_fraction=float(np.mean((a < 0) | (a > 1)))) for a in outputs])
