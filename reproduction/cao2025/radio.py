"""Explicit exploratory air protocol; protected boxes are never free side data.

One frame: optional pilots, protected fixed prefix, protected box body, analog
data. Receiver functions take received symbols and public channel settings only.
This repetition/CRC format is a documented choice, not Cao's specified code.
"""
from dataclasses import dataclass
import math
import struct
import zlib
import torch
import torch.nn.functional as F

PREFIX = struct.Struct('!4sBHHHH')
BOX = struct.Struct('!HHHHf')
REPETITION = 7
PREFIX_USES = (PREFIX.size + 4)*8*REPETITION
MAX_BOXES = 300
MAX_DIMENSION = 8192


class FrameErasure(ValueError):
    """Received control or length failed validation; no clean-control fallback."""


@dataclass(frozen=True)
class Observation:
    symbols: torch.Tensor  # Nx2 real/imaginary, including received pilots
    noise_power: float
    channel: str


def checked_boxes(shape, views):
    h, w = shape
    if not (192 < h <= MAX_DIMENSION and 192 < w <= MAX_DIMENSION):
        raise ValueError('unsupported native stereo shape')
    if len(views) != 2 or any(len(boxes) > MAX_BOXES for boxes in views):
        raise ValueError('exactly two bounded box lists required')
    for boxes in views:
        for box in boxes:
            coords = box['xyxy']
            if len(coords) != 4 or any(int(x) != x for x in coords):
                raise ValueError('integer native box coordinates required')
            x1, y1, x2, y2 = coords
            if not (0 <= x1 <= x2 <= w and 0 <= y1 <= y2 <= h):
                raise ValueError('box outside native image')
            if not math.isfinite(box['confidence']) or not 0 <= box['confidence'] <= 1:
                raise ValueError('invalid box confidence')


def crc_bytes(payload):
    return payload + struct.pack('!I', zlib.crc32(payload))


def bpsk_bytes(payload, like):
    bits = [(byte >> shift) & 1 for byte in payload for shift in range(7, -1, -1)]
    real = (torch.tensor(bits, device=like.device, dtype=like.dtype)*2-1).repeat_interleave(REPETITION)
    return torch.stack((real, torch.zeros_like(real)), -1)


def read_bpsk(symbols, nbytes):
    if symbols.shape != (nbytes*8*REPETITION, 2):
        raise FrameErasure('protected control length mismatch')
    bits = (symbols[:, 0].reshape(-1, REPETITION).sum(1) > 0).cpu().tolist()
    raw = bytes(sum(int(bits[i+j]) << (7-j) for j in range(8)) for i in range(0, len(bits), 8))
    if len(raw) < 4 or zlib.crc32(raw[:-4]) != struct.unpack('!I', raw[-4:])[0]:
        raise FrameErasure('control CRC failed')
    return raw[:-4]


def encode_control(shape, views, like):
    checked_boxes(shape, views)
    prefix = PREFIX.pack(b'CS25', 1, *shape, *(len(boxes) for boxes in views))
    body = b''.join(BOX.pack(*box['xyxy'], box['confidence']) for boxes in views for box in boxes)
    return torch.cat((bpsk_bytes(crc_bytes(prefix), like), bpsk_bytes(crc_bytes(body), like)))


def decode_control(symbols):
    prefix = read_bpsk(symbols[:PREFIX_USES], PREFIX.size+4)
    magic, version, h, w, nl, nr = PREFIX.unpack(prefix)
    if magic != b'CS25' or version != 1 or max(nl, nr) > MAX_BOXES:
        raise FrameErasure('unsupported control format/counts')
    # Validate bounds before consuming the variable body or allocating masks.
    if not (192 < h <= MAX_DIMENSION and 192 < w <= MAX_DIMENSION):
        raise FrameErasure('unsupported received shape')
    body_bytes = (nl+nr)*BOX.size+4
    end = PREFIX_USES+body_bytes*8*REPETITION
    body = read_bpsk(symbols[PREFIX_USES:end], body_bytes)
    boxes = []
    for offset in range(0, len(body), BOX.size):
        *coords, confidence = BOX.unpack_from(body, offset)
        boxes.append(dict(xyxy=coords, confidence=confidence))
    views = (boxes[:nl], boxes[nl:])
    try:
        checked_boxes((h, w), views)
    except ValueError as error:
        raise FrameErasure(str(error)) from error
    return (h, w), views, end


def roi_masks(shape, views, like):
    checked_boxes(shape, views)
    h, w = shape
    masks = []
    for boxes in views:
        mask = like.new_zeros((1, 1, h, w))
        for box in boxes:
            x1, y1, x2, y2 = box['xyxy']
            mask[..., y1:y2, x1:x2] = 1
        masks.append(F.pad(mask, (0, (-w)%6, 0, (-h)%6)))
    return tuple(masks)


def support_masks(masks):
    # A key code cell is sent iff at least one original ROI pixel projects to it.
    return tuple(F.max_pool2d(mask, 2, 2).bool() for mask in masks)


def normalize_data(real):
    if real.ndim != 1 or real.numel() == 0 or not torch.isfinite(real).all():
        raise ValueError('nonempty finite real stream required')
    paired = F.pad(real, (0, real.numel()%2)).reshape(-1, 2)
    energy = paired.square().sum()
    if not torch.isfinite(energy) or energy <= 0:
        raise ValueError('zero/nonfinite transmitted energy')
    # The factor is solely a transmitter operation. It is never sent to decoder.
    return paired * torch.sqrt(energy.new_tensor(paired.shape[0])/energy)


def pilot_count(channel):
    if channel not in ('identity', 'awgn', 'rayleigh'):
        raise ValueError('unsupported channel')
    return 8 if channel == 'rayleigh' else 0


def propagate(symbols, channel, snr_db, generator=None):
    if symbols.ndim != 2 or symbols.shape[1] != 2 or not torch.isfinite(symbols).all():
        raise ValueError('finite complex symbols required')
    pilots = pilot_count(channel)
    noise_power = 0. if channel == 'identity' else 10.**(-float(snr_db)/10.)
    if not math.isfinite(noise_power) or noise_power < 0:
        raise ValueError('invalid noise power')
    if channel == 'rayleigh':
        h = torch.randn(2, device=symbols.device, dtype=symbols.dtype, generator=generator)/math.sqrt(2)
        real, imag = symbols.unbind(-1)
        result = torch.stack((real*h[0]-imag*h[1], real*h[1]+imag*h[0]), -1)
        if symbols.shape[0] <= pilots or not torch.equal(symbols[:pilots], symbols.new_tensor([1., 0.]).expand(pilots, 2)):
            raise ValueError('explicit unit pilots required for fading')
    else:
        result = symbols
    if noise_power:
        result = result + torch.randn(result.shape, device=result.device, dtype=result.dtype,
            generator=generator)*math.sqrt(noise_power/2)
    return Observation(result, noise_power, channel)


def equalize(observation):
    symbols = observation.symbols
    count = pilot_count(observation.channel)
    if symbols.ndim != 2 or symbols.shape[1] != 2 or not torch.isfinite(symbols).all():
        raise FrameErasure('invalid received symbols')
    if not math.isfinite(observation.noise_power) or observation.noise_power < 0:
        raise FrameErasure('invalid public noise power')
    if not count:
        return symbols
    if len(symbols) <= count:
        raise FrameErasure('missing received pilots/data')
    # Pilot-only LMMSE channel estimation. No access to the actual fading h.
    estimate = symbols[:count].mean(0)/(1+observation.noise_power/count)
    denominator = estimate.square().sum()+observation.noise_power
    if denominator <= 0:
        raise FrameErasure('degenerate pilot estimate')
    real, imag = symbols[count:].unbind(-1)
    return torch.stack((real*estimate[0]+imag*estimate[1],
                        imag*estimate[0]-real*estimate[1]), -1)/denominator
