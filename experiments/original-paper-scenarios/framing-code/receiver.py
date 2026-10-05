"""Receiver sees decoded information blocks and public code k only."""
import hashlib
from pathlib import Path
import struct
import sys
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'experiments/original-paper-scenarios/code'))
from image_codecs import decode_pair

HEADER = struct.Struct('>4sBBHIII')


class FramingError(ValueError):
    pass


def need(ok, reason):
    if not ok:
        raise FramingError(reason)


def extract_wire(decoded, *, k):
    """No transmitter length/bytes are arguments. Non-byte tails stay as bits."""
    need(type(k) is int and k in (1296, 972), 'unsupported_public_k')
    a = np.asarray(decoded)
    need(a.ndim == 2 and a.shape[0] > 0 and a.shape[1] == k, 'decoded_block_shape')
    need(a.dtype.kind in 'biuf' and np.isfinite(a).all() and ((a == 0) | (a == 1)).all(),
         'nonbinary_decoded_bits')
    bits = a.astype(np.uint8, copy=False).reshape(-1)
    need(len(bits) >= 8 * HEADER.size, 'truncated_header')
    head = np.packbits(bits[:8 * HEADER.size], bitorder='big').tobytes()
    magic, version, codec, reserved, nl, nr, crc = HEADER.unpack(head)
    need(magic == b'P6SB', 'header_magic')
    need(version == 1, 'header_version')
    need(codec in (1, 2), 'header_codec')
    need(reserved == 0, 'header_reserved')
    need(nl > 0 and nr > 0, 'header_zero_length')
    nbytes = HEADER.size + int(nl) + int(nr)
    nbits = 8 * nbytes
    need(nbits <= len(bits), 'header_capacity_overflow')
    expected_blocks = (nbits + k - 1) // k
    need(expected_blocks == a.shape[0], 'observed_block_count_mismatch')
    need(not np.any(bits[nbits:]), 'nonzero_decoded_padding')
    wire = np.packbits(bits[:nbits], bitorder='big').tobytes()
    need(zlib.crc32(wire[HEADER.size:]) == crc, 'payload_CRC_failure')
    evidence = dict(receiver_inputs=['decoded_information_blocks', 'public_code_k'],
                    observed_blocks=int(a.shape[0]), public_k=k, header_bytes=HEADER.size,
                    decoded_header_hex=head.hex(), header_derived_bytes=nbytes,
                    source_padding_bits=len(bits) - nbits, payload_lengths=[int(nl), int(nr)],
                    codec='jpeg' if codec == 1 else 'jpeg2000',
                    decoded_wire_sha256=hashlib.sha256(wire).hexdigest(),
                    source_length_side_channel=False)
    return wire, evidence


def receive_stereo(decoded, *, k):
    """Fail closed: return no image or wire on any failed reception."""
    record = dict(state='erased', receiver_inputs=['decoded_information_blocks', 'public_code_k'],
                  source_length_side_channel=False, erasure_reason=None)
    try:
        wire, metadata = extract_wire(decoded, k=k)
        record.update(metadata)
    except FramingError as exc:
        record['erasure_reason'] = str(exc)
        return None, None, record
    try:
        images = decode_pair(wire)
    except Exception as exc:
        record.update(erasure_reason='image_decode_or_schema_failure',
                      image_error_class=type(exc).__name__)
        return None, None, record
    record.update(state='received', native_shape=list(images[0].shape), erasure_reason=None,
                  received_image_sha256=[hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
                                         for a in images])
    return wire, images, record
