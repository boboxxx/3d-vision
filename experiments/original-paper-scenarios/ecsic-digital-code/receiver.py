"""P6SB v2 receiver: decoded information blocks and public k only."""
import hashlib
import importlib.util
import struct
import zlib

import numpy as np

from common import ROOT, CDF_SHA, need, sha

ENTROPY_PATH = ROOT / 'experiments/original-paper-scenarios/ecsic-entropy-code/codec.py'
ENTROPY_SHA = 'a22fb7717cd1e2a13e4539a74a541958e93a7f6b215ae960b9c2736ad53ec157'
HEADER = struct.Struct('>4sBBHIII')
LIMIT = 32 * 1024 * 1024
MIN_CONTAINER = 182


def entropy_codec():
    # Dependency failure is fatal, outside packet-malformation handling.
    need(sha(ENTROPY_PATH) == ENTROPY_SHA, 'sealed entropy parser changed')
    spec = importlib.util.spec_from_file_location('ecsic_digital_sealed_entropy', ENTROPY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, digest = module.tables()
    need(digest == CDF_SHA, 'sealed CDF changed')
    return module


class FrameError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise FrameError(message)


def wrap_container(payload):
    codec = entropy_codec()
    require(type(payload) is bytes, 'immutable_P6EC_bytes_required')
    codec.unpack(payload, CDF_SHA)
    return HEADER.pack(b'P6SB', 2, 3, 0, len(payload), 0, zlib.crc32(payload)) + payload


def extract_container(decoded, *, k, codec):
    require(type(k) is int and k in (1296, 972), 'unsupported_public_k')
    a = np.asarray(decoded)
    require(a.ndim == 2 and a.shape[0] > 0 and a.shape[1] == k, 'decoded_block_shape')
    require(a.shape[0] <= (8*(LIMIT+HEADER.size)+k-1)//k, 'observed_capacity_limit')
    require(a.dtype.kind in 'biuf' and np.isfinite(a).all() and ((a == 0) | (a == 1)).all(), 'nonbinary_decoded_bits')
    bits = a.astype(np.uint8, copy=False).reshape(-1)
    require(bits.size >= HEADER.size*8, 'truncated_header')
    head = np.packbits(bits[:HEADER.size*8], bitorder='big').tobytes()
    magic, version, tag, reserved, length, second, crc = HEADER.unpack(head)
    require(magic == b'P6SB', 'header_magic')
    require(version == 2, 'header_version')
    require(tag == 3, 'header_codec')
    require(reserved == 0, 'header_reserved')
    require(MIN_CONTAINER <= length <= LIMIT, 'header_payload_length')
    require(second == 0, 'header_second_length')
    nbytes = HEADER.size + length
    nbits = nbytes * 8
    require(nbits <= bits.size, 'header_capacity_overflow')
    require((nbits+k-1)//k == a.shape[0], 'observed_block_count_mismatch')
    require(not bits[nbits:].any(), 'nonzero_decoded_padding')
    wire = np.packbits(bits[:nbits], bitorder='big').tobytes()
    payload = wire[HEADER.size:]
    require(zlib.crc32(payload) == crc, 'outer_payload_CRC_failure')
    try:
        parsed = codec.unpack(payload, CDF_SHA)
    except ValueError as exc:
        raise FrameError('inner_P6EC_failure:' + str(exc)) from exc
    return wire, payload, dict(observed_blocks=int(a.shape[0]), public_k=k, decoded_header_hex=head.hex(),
                              header_bytes=HEADER.size, header_derived_wire_bytes=nbytes,
                              header_derived_payload_bytes=length, source_padding_bits=int(bits.size-nbits),
                              codec='ecsic_P6EC', outer_version=2, payload_lengths=[length, 0],
                              original_hw=parsed['original_hw'], padded_hw=parsed['padded_hw'],
                              inner_overhead_bytes=parsed['overhead_bytes'],
                              received_wire_sha256=hashlib.sha256(wire).hexdigest(),
                              received_payload_sha256=hashlib.sha256(payload).hexdigest())


def receive_container(decoded, *, k):
    codec = entropy_codec()
    record = dict(state='erased', receiver_inputs=['decoded_information_blocks', 'public_code_k'],
                  source_length_side_channel=False, erasure_reason=None, neural_decoder_called=False)
    try:
        wire, payload, metadata = extract_container(decoded, k=k, codec=codec)
    except FrameError as exc:
        record['erasure_reason'] = str(exc)
        return None, None, record
    record.update(metadata, state='received')
    return wire, payload, record
