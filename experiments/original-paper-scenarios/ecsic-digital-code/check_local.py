"""Actual P6EC framing fixtures and a separately written byte-level parser."""
import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import struct
import time
import traceback
from unittest.mock import patch
import zlib

import numpy as np

from common import ROOT, PROTOCOL_SHA, CDF_SHA, CONFIGS, SOURCES, identities, source_payloads, packet_plan, save, sha, need
from receiver import HEADER, LIMIT, receive_container, wrap_container


def independent_inner(payload):
    if not 182 <= len(payload) <= 33554432:
        return False
    if zlib.crc32(payload[:-4]) != int.from_bytes(payload[-4:], 'big') or payload[:6] != b'P6EC\x01\x04':
        return False
    dims = [int.from_bytes(payload[p:p+2], 'big') for p in (6, 8, 10, 12)]
    h, w, ph, pw = dims
    if not (0 < h <= ph <= 2048 and 0 < w <= pw <= 4096 and ph % 32 == pw % 32 == 0 and ph-h < 32 and pw-w < 32):
        return False
    expected = ('e66b55e2f888f5d113ea6c6414640c7486280d6e18838ba77c650720a5636052',
                'ba02cc1239b755a26ca0eb920c3014908c97c45fa210c447c47e96f7527102f8', CDF_SHA)
    if tuple(payload[p:p+32].hex() for p in (14, 46, 78)) != expected:
        return False
    end = 162
    for index in range(4):
        p = 110 + index*13
        sid = payload[p]
        count, nr, ne = [int.from_bytes(payload[q:q+4], 'big') for q in (p+1, p+5, p+9)]
        divisor = 32 if index < 2 else 8
        wanted = 48*(ph//divisor)*(pw//divisor)
        if sid != index or count != wanted or nr < 4 or ne > 4*wanted or end+nr+ne > len(payload)-4:
            return False
        end += nr+ne
    return end == len(payload)-4


def independent_received(a, k):
    if type(k) is not int or k not in (1296, 972):
        return 'unsupported_public_k', None
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] != k:
        return 'decoded_block_shape', None
    if a.shape[0] > (8*(33554432+20)+k-1)//k:
        return 'observed_capacity_limit', None
    if a.dtype.kind not in 'biuf' or not np.isfinite(a).all() or not np.isin(a, [0, 1]).all():
        return 'nonbinary_decoded_bits', None
    b = a.astype(np.uint8).ravel()
    head = np.packbits(b[:160], bitorder='big').tobytes()
    for ok, why in [(head[:4] == b'P6SB', 'header_magic'), (head[4] == 2, 'header_version'),
                    (head[5] == 3, 'header_codec'), (head[6:8] == b'\0\0', 'header_reserved')]:
        if not ok:
            return why, None
    length, second, crc = [int.from_bytes(head[p:p+4], 'big') for p in (8, 12, 16)]
    if not 182 <= length <= 33554432:
        return 'header_payload_length', None
    if second:
        return 'header_second_length', None
    end = 8*(20+length)
    if end > b.size:
        return 'header_capacity_overflow', None
    if (end+k-1)//k != len(a):
        return 'observed_block_count_mismatch', None
    if b[end:].any():
        return 'nonzero_decoded_padding', None
    payload = np.packbits(b[160:end], bitorder='big').tobytes()
    if zlib.crc32(payload) != crc:
        return 'outer_payload_CRC_failure', None
    if not independent_inner(payload):
        return 'inner_P6EC_failure', None
    return None, payload


def blocks(wire, k):
    bits = np.unpackbits(np.frombuffer(wire, dtype=np.uint8), bitorder='big')
    return np.pad(bits, (0, (-len(bits)) % k)).reshape(-1, k)


def outer_unchecked(payload):
    return b'P6SB\x02\x03\0\0' + len(payload).to_bytes(4, 'big') + bytes(4) + zlib.crc32(payload).to_bytes(4, 'big') + payload


def changed_outer(wire, field, value):
    values = list(HEADER.unpack_from(wire)); values[field] = value
    return HEADER.pack(*values) + wire[20:]


def changed_inner(payload, offset, replacement, recompute=True):
    value = bytearray(payload)
    value[offset:offset+len(replacement)] = replacement
    if recompute:
        value[-4:] = zlib.crc32(value[:-4]).to_bytes(4, 'big')
    return outer_unchecked(bytes(value))


def run(output):
    output = Path(output).resolve()
    artifacts = output.with_suffix('.artifacts')
    need(not output.exists() and not artifacts.exists(), 'unique check paths required')
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir()
    record = dict(state='running', scope='P6EC_outer_framing_CPU_no_PHY_no_AP', started_unix=time.time(),
                  protocol_sha256=PROTOCOL_SHA, numpy=np.__version__, checks=[])
    save(output, record)
    fixtures = {}
    rng = copy.deepcopy(np.random.get_state())
    try:
        before = identities()
        payloads = source_payloads()
        record['source_sha256'] = before
        need(list(inspect.signature(receive_container).parameters) == ['decoded', 'k'], 'receiver input API')
        plan = packet_plan()
        need(len(plan) == 20 and len({p['name'] for p in plan}) == 20, 'complete unique packet matrix')
        need([p['noise_seed'] for p in plan] == list(range(1911, 1951, 2)) and
             [p['fading_seed'] for p in plan] == list(range(1912, 1952, 2)), 'locked seeds')
        record['packet_plan'] = plan
        for source, payload in payloads.items():
            wire = wrap_container(payload)
            need(wire == outer_unchecked(payload), 'independent outer serialization')
            (artifacts / (source+'.p6sb')).write_bytes(wire)
            for configuration, (k, n, bps) in CONFIGS.items():
                base = blocks(wire, k)
                stem = source+'-'+str(k)
                cases = [('valid', base, k, None)]
                outer_fields = [('magic', 0, b'FAIL', 'header_magic'), ('version', 1, 1, 'header_version'),
                                ('codec', 2, 2, 'header_codec'), ('reserved', 3, 1, 'header_reserved'),
                                ('zero_length', 4, 0, 'header_payload_length'),
                                ('short_length', 4, 181, 'header_payload_length'),
                                ('oversize', 4, LIMIT+1, 'header_payload_length'),
                                ('capacity', 4, len(payload)+k, 'header_capacity_overflow'),
                                ('second_length', 5, 1, 'header_second_length'),
                                ('outer_crc', 6, zlib.crc32(payload)^1, 'outer_payload_CRC_failure')]
                cases.extend((name, blocks(changed_outer(wire, field, value), k), k, reason)
                             for name, field, value, reason in outer_fields)
                cases += [('missing_block', base[:-1], k, 'header_capacity_overflow'),
                          ('extra_block', np.vstack((base, np.zeros((1, k), dtype=np.uint8))), k, 'observed_block_count_mismatch'),
                          ('empty', base[:0], k, 'decoded_block_shape'),
                          ('wrong_width', base[:, :-1], k, 'decoded_block_shape'),
                          ('wrong_k', base, 111, 'unsupported_public_k')]
                for name, value in [('nonbinary', .5), ('nan', np.nan), ('inf', np.inf)]:
                    damaged = base.astype(np.float64); damaged[0, 0] = value
                    cases.append((name, damaged, k, 'nonbinary_decoded_bits'))
                for name, position in [('first_pad_bit', len(wire)*8), ('last_pad_bit', base.size-1)]:
                    need(position >= len(wire)*8, 'fixed fixture has padding')
                    damaged = base.copy(); damaged.ravel()[position] = 1
                    cases.append((name, damaged, k, 'nonzero_decoded_padding'))
                damaged = base.copy(); damaged.ravel()[160] ^= 1
                cases.append(('payload_bit', damaged, k, 'outer_payload_CRC_failure'))
                inner_cases = [('inner_crc', 0, b'X', False), ('inner_magic', 0, b'FAIL', True),
                               ('inner_version', 4, b'\x02', True), ('inner_count', 5, b'\x03', True),
                               ('inner_model', 14, bytes(32), True), ('inner_config', 46, bytes(32), True),
                               ('inner_cdf', 78, bytes(32), True), ('inner_zero_height', 6, bytes(2), True),
                               ('inner_padding', 10, b'\x00\x21', True), ('inner_order', 110, b'\x01', True),
                               ('inner_symbols', 111, bytes(4), True), ('inner_rans_length', 115, bytes(4), True),
                               ('inner_escape_length', 119, b'\xff'*4, True)]
                cases.extend((name, blocks(changed_inner(payload, offset, value, repair), k), k, 'inner_P6EC_failure')
                             for name, offset, value, repair in inner_cases)
                for name, inner in [('inner_append', payload[:-4]+b'X'), ('inner_truncate', payload[:-5])]:
                    inner += zlib.crc32(inner).to_bytes(4, 'big')
                    cases.append((name, blocks(outer_unchecked(inner), k), k, 'inner_P6EC_failure'))
                for name, fixture, public_k, reason in cases:
                    w, got, received = receive_container(fixture, k=public_k)
                    independently, ipayload = independent_received(fixture, public_k)
                    actual = received['erasure_reason']
                    need(independently == reason, 'independent fixture diagnosis: '+name)
                    if reason is None:
                        need(w == wire and got == ipayload == payload and received['state'] == 'received', 'actual intact container')
                        need(received['header_derived_wire_bytes'] == len(wire), 'received header length')
                        need(received['source_padding_bits'] == base.size-len(wire)*8, 'whole-block source padding')
                    else:
                        need(w is None and got is None and received['state'] == 'erased', 'erasure has no payload fallback')
                        need(actual == reason or (reason == 'inner_P6EC_failure' and actual.startswith(reason+':')), 'specific failure reason: '+name)
                    key = stem+'-'+name
                    fixtures[key] = fixture
                    record['checks'].append(dict(name=key, public_k=public_k, expected_reason=reason,
                                                 reception=received, blocks=int(fixture.shape[0])))
        with patch('receiver.entropy_codec', side_effect=RuntimeError('injected_dependency_failure')):
            try:
                receive_container(base, k=k)
            except RuntimeError as exc:
                need(str(exc) == 'injected_dependency_failure', 'dependency error was not propagated')
            else:
                raise AssertionError('dependency failure converted to erasure')
        record['fatal_dependency_failure_propagated'] = True
        need(identities() == before, 'historical/new source changed during checks')
        need(all(np.array_equal(a, b) for a, b in zip(rng, np.random.get_state())), 'global RNG changed')
        np.savez_compressed(artifacts/'decoded-fixtures.npz', **fixtures)
        record.update(state='passed_independent_P6EC_framing_checks', checks_count=len(record['checks']),
                      successful_fixtures=sum(c['expected_reason'] is None for c in record['checks']),
                      global_numpy_rng_unchanged=True, source_after_sha256=identities(),
                      artifact_sha256={str(p.relative_to(ROOT)): sha(p) for p in artifacts.iterdir()},
                      physical_channel_executed=False, neural_decoder_executed=False)
    except BaseException:
        record.update(state='failed', traceback=traceback.format_exc())
        raise
    finally:
        record['finished_unix'] = time.time()
        save(output, record)
    print(json.dumps({k: record[k] for k in ('state', 'checks_count', 'successful_fixtures')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
