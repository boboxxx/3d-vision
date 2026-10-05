"""Actual fixed-parameter source bitstreams. No rate targeting or GT selection."""
import hashlib
import io
import math
import struct
import zlib

import numpy as np
from PIL import Image

from contracts import digest_array, require

HEADER = struct.Struct(">4sBBHIII")
MAGIC = b"P6SB"
CODECS = {"jpeg": 1, "jpeg2000": 2}


def rgb(value):
    value = np.asarray(value)
    require(value.dtype == np.uint8 and value.ndim == 3 and value.shape[2] == 3,
            "native HxWx3 uint8 RGB required")
    require(value.shape[0] > 0 and value.shape[1] > 0, "empty image")
    return value


def encode_pair(left, right, *, codec, parameter):
    left, right = rgb(left), rgb(right)
    require(left.shape == right.shape and codec in CODECS, "matched pair/known codec")
    if codec == "jpeg":
        require(type(parameter) is int and 1 <= parameter <= 95, "JPEG quality1–95")
        kwargs = {"quality": parameter, "subsampling": 2, "optimize": False, "progressive": False}
        fmt = "JPEG"
    else:
        require(type(parameter) in (int, float) and math.isfinite(parameter) and parameter >= 1,
                "JP2 rates>=1")
        kwargs = {"quality_mode": "rates", "quality_layers": [parameter],
                  "irreversible": True, "mct": 1, "no_jp2": False}
        fmt = "JPEG2000"
    streams = []
    for array in (left, right):
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format=fmt, **kwargs)
        streams.append(buffer.getvalue())
    require(all(0 < len(v) <= 0xffffffff for v in streams), "invalid stream size")
    payload = b"".join(streams)
    header = HEADER.pack(MAGIC, 1, CODECS[codec], 0, len(streams[0]), len(streams[1]),
                         zlib.crc32(payload))
    wire = header + payload
    decoded = decode_pair(wire)
    require(decoded[0].shape == decoded[1].shape == left.shape, "native geometry changed")
    evidence = {
        "codec": codec, "parameter": parameter, "options": kwargs,
        "left_input": digest_array(left), "right_input": digest_array(right),
        "left_received": digest_array(decoded[0]), "right_received": digest_array(decoded[1]),
        "raw_RGB8_bits": int((left.size + right.size) * 8),
        "codestream_bytes": [len(v) for v in streams], "framing_bytes": HEADER.size,
        "source_bytes": len(wire), "source_bits": 8 * len(wire),
        "actual_raw_to_serialized_ratio": (left.size + right.size) / len(wire),
        "wire_sha256": hashlib.sha256(wire).hexdigest(),
    }
    return wire, decoded, evidence


def decode_pair(wire):
    require(type(wire) is bytes and len(wire) > HEADER.size, "complete immutable bitstream")
    magic, version, codec_id, reserved, nl, nr, crc = HEADER.unpack_from(wire)
    require(magic == MAGIC and version == 1 and codec_id in CODECS.values() and reserved == 0,
            "invalid framing")
    require(nl > 0 and nr > 0 and len(wire) == HEADER.size + nl + nr, "stream length mismatch")
    payload = wire[HEADER.size:]
    require(zlib.crc32(payload) == crc, "CRC failure: erase entire stereo frame")
    expected_format = "JPEG" if codec_id == 1 else "JPEG2000"
    arrays = []
    for stream in (payload[:nl], payload[nl:]):
        with Image.open(io.BytesIO(stream)) as image:
            require(image.format == expected_format and image.mode == "RGB", "codec/schema mismatch")
            arrays.append(np.array(image, dtype=np.uint8))
    require(arrays[0].shape == arrays[1].shape, "stereo geometry mismatch")
    return tuple(arrays)
