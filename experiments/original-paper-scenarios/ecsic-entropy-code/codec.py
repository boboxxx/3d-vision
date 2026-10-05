"""Declared finite-CDF ECSIC entropy container; no neural/source-input access."""
from bisect import bisect_right
import hashlib
from pathlib import Path
import struct
import zlib

import numpy as np

HERE = Path(__file__).resolve().parent
L = 1 << 23
TOTAL = 1 << 16
ORDER = ('z_left', 'z_right', 'y_left', 'y_right')
MODEL_SHA = 'e66b55e2f888f5d113ea6c6414640c7486280d6e18838ba77c650720a5636052'
CONFIG_SHA = 'ba02cc1239b755a26ca0eb920c3014908c97c45fa210c447c47e96f7527102f8'
BASE = struct.Struct('>4sBB4H32s32s32s')
DESC = struct.Struct('>BIII')
LIMIT = 32 * 1024 * 1024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def make_tables():
    """Run once before measurement; persisted integer table defines the codec."""
    scales = np.exp(np.linspace(np.log(.1), np.log(256.), 256, dtype=np.float64))
    scales[0], scales[-1] = .1, 256.
    k = np.abs(np.arange(-255, 256, dtype=np.float64))
    rows = []
    for scale in scales:
        p = .5 * (np.exp(-np.maximum(k-.5, 0)/scale) - np.exp(-(k+.5)/scale))
        p[255] = -np.expm1(-.5 / scale)
        p = np.append(p, np.exp(-255.5 / scale))
        p = np.maximum(p, 0); p /= p.sum()
        weighted = p * (TOTAL - 512)
        floor = np.floor(weighted).astype(np.int64)
        freq = floor + 1
        remainder = TOTAL - int(freq.sum())
        require(0 <= remainder < 512, 'CDF remainder')
        freq[np.argsort(-(weighted-floor), kind='stable')[:remainder]] += 1
        rows.append(np.concatenate(([0], np.cumsum(freq))))
    table = np.asarray(rows, dtype='<u4')
    require(table.shape == (256, 513) and np.all(np.diff(table.astype(np.int64), axis=1) > 0), 'CDF support')
    require(np.all(table[:, -1] == TOTAL), 'CDF total')
    return table


def tables():
    blob = (HERE / 'cdf-u32le.bin').read_bytes()
    expected = (HERE / 'cdf.sha256').read_text().strip()
    require(hashlib.sha256(blob).hexdigest() == expected, 'sealed CDF identity')
    table = np.frombuffer(blob, dtype='<u4').reshape(256, 513)
    require(np.all(table[:, 0] == 0) and np.all(table[:, -1] == TOTAL), 'CDF endpoints')
    require(np.all(np.diff(table.astype(np.int64), axis=1) > 0), 'CDF strictly positive')
    return tuple(tuple(map(int, row)) for row in table), expected


def scale_ids(scale, shape):
    a = np.broadcast_to(np.asarray(scale, dtype=np.float64), shape)
    require(np.isfinite(a).all(), 'finite probability scale')
    log_index = (np.log(np.clip(a, .1, 256.)) - np.log(.1)) * (255 / np.log(2560.))
    return np.clip(np.rint(log_index), 0, 255).astype(np.uint8).ravel(order='C')


def rans_encode(symbols, indices, cdfs):
    require(len(symbols) == len(indices) and len(symbols) > 0, 'symbol/table count')
    x, emitted = L, bytearray()
    for symbol, index in zip(reversed(symbols), reversed(indices)):
        symbol, index = int(symbol), int(index)
        require(0 <= symbol < 512 and 0 <= index < 256, 'symbol/table bounds')
        row = cdfs[index]; start = row[symbol]; freq = row[symbol+1]-start
        maximum = ((L >> 16) << 8) * freq
        while x >= maximum:
            emitted.append(x & 255); x >>= 8
        x = ((x // freq) << 16) + x % freq + start
    return struct.pack('<I', x) + bytes(reversed(emitted))


def rans_decode(blob, indices, cdfs):
    require(len(blob) >= 4 and len(indices) > 0, 'nonempty rANS')
    x = struct.unpack_from('<I', blob)[0]; position = 4
    require(L <= x < 2**31, 'initial rANS state')
    result = np.empty(len(indices), dtype=np.int32)
    for i, index in enumerate(indices):
        require(0 <= int(index) < 256, 'table index')
        row = cdfs[int(index)]; cumulative = x & (TOTAL-1)
        symbol = bisect_right(row, cumulative) - 1
        start = row[symbol]; freq = row[symbol+1] - start
        result[i] = symbol
        x = freq * (x >> 16) + cumulative - start
        while x < L:
            require(position < len(blob), 'truncated rANS')
            x = (x << 8) | blob[position]; position += 1
    require(x == L and position == len(blob), 'rANS terminal state/unused bytes')
    return result


def varint(value):
    require(abs(value) < 2**24, 'escape residual range')
    u = 2*value if value >= 0 else -2*value-1
    out = bytearray()
    while u >= 128:
        out.append((u & 127) | 128); u >>= 7
    out.append(u)
    return bytes(out)


def read_varint(blob, position):
    start, value = position, 0
    for shift in range(0, 28, 7):
        require(position < len(blob), 'truncated escape')
        b = blob[position]; position += 1
        value |= (b & 127) << shift
        if not b & 128:
            signed = -(value // 2)-1 if value & 1 else value // 2
            require(abs(signed) < 2**24 and not -255 <= signed <= 255, 'escape support')
            require(blob[start:position] == varint(signed), 'noncanonical escape')
            return signed, position
    raise ValueError('oversized escape')


def encode_stream(residual, scales, cdfs):
    a = np.asarray(residual)
    require(a.dtype == np.int32 and a.size > 0 and np.max(np.abs(a.astype(np.int64))) < 2**24, 'residual integer range')
    values = a.ravel(order='C')
    mask = (values < -255) | (values > 255)
    symbols = np.where(mask, 511, values+255)
    escapes = b''.join(varint(int(v)) for v in values[mask])
    return rans_encode(symbols, scale_ids(scales, a.shape), cdfs), escapes


def decode_stream(rans, escapes, scales, shape, cdfs):
    coded = rans_decode(rans, scale_ids(scales, shape), cdfs)
    values = coded - 255; position = 0
    for i in np.flatnonzero(coded == 511):
        values[i], position = read_varint(escapes, position)
    require(position == len(escapes), 'unused escape bytes')
    return values.reshape(shape)


def dimensions(original, padded):
    require(len(original) == len(padded) == 2, 'dimension fields')
    h, w = original; ph, pw = padded
    require(all(type(x) is int for x in (h, w, ph, pw)), 'integer dimensions')
    require(0 < h <= ph <= 2048 and 0 < w <= pw <= 4096, 'dimension bounds')
    require(ph % 32 == pw % 32 == 0 and ph-h < 32 and pw-w < 32, 'padding rule')


def shape_of(index, padded):
    h, w = padded; divisor = 32 if index < 2 else 8
    return 1, 48, h // divisor, w // divisor


def pack(original, padded, streams, cdf_sha):
    dimensions(original, padded)
    require(list(streams) == list(ORDER), 'four-stream order')
    header = bytearray(BASE.pack(b'P6EC', 1, 4, *original, *padded,
                                bytes.fromhex(MODEL_SHA), bytes.fromhex(CONFIG_SHA), bytes.fromhex(cdf_sha)))
    body = bytearray()
    for index, name in enumerate(ORDER):
        rans, escapes = streams[name]
        require(len(rans) >= 4 and len(rans) + len(escapes) <= LIMIT, 'stream byte limit')
        header.extend(DESC.pack(index, int(np.prod(shape_of(index, padded))), len(rans), len(escapes)))
        body.extend(rans); body.extend(escapes)
    out = bytes(header + body)
    require(len(out)+4 <= LIMIT, 'container byte limit')
    return out + struct.pack('>I', zlib.crc32(out))


def unpack(blob, cdf_sha):
    require(166+16 <= len(blob) <= LIMIT, 'container length bound')
    require(zlib.crc32(blob[:-4]) == struct.unpack('>I', blob[-4:])[0], 'container CRC')
    magic, version, count, h, w, ph, pw, model, config, table = BASE.unpack_from(blob)
    require((magic, version, count) == (b'P6EC', 1, 4), 'container format')
    require((model.hex(), config.hex(), table.hex()) == (MODEL_SHA, CONFIG_SHA, cdf_sha), 'container model/config/CDF identity')
    dimensions([h, w], [ph, pw])
    offset = BASE.size + 4*DESC.size; streams = {}
    for index, name in enumerate(ORDER):
        sid, count, nr, ne = DESC.unpack_from(blob, BASE.size+index*DESC.size)
        expected = int(np.prod(shape_of(index, [ph, pw])))
        require(sid == index and count == expected, 'stream id/count')
        require(nr >= 4 and ne <= 4*expected and offset + nr + ne <= len(blob)-4, 'stream lengths')
        streams[name] = (blob[offset:offset+nr], blob[offset+nr:offset+nr+ne])
        offset += nr + ne
    require(offset == len(blob)-4, 'unused container bytes')
    return dict(original_hw=[h, w], padded_hw=[ph, pw], streams=streams,
                total_bytes=len(blob), overhead_bytes=166)
