"""Meaningful real-codec and malformed-frame receiver checks; no LDPC runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zlib

import numpy as np
from receiver import ROOT, HEADER, extract_wire, receive_stereo
from image_codecs import encode_pair


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def source_identities():
    return {str(p.relative_to(ROOT)): sha(p) for d in (
        Path(__file__).parent, ROOT/'experiments/original-paper-scenarios/code',
        ROOT/'experiments/original-paper-scenarios/digital-code')
        for p in sorted(d.glob('*.py'))}


def pair():
    y, x, c = np.indices((8, 12, 3))
    left = ((y*31 + x*17 + c*73) % 256).astype(np.uint8)
    return left, np.roll(left, 1, axis=1)


def blocks(wire, k):
    bits = np.unpackbits(np.frombuffer(wire, dtype=np.uint8), bitorder='big')
    return np.pad(bits, (0, (-len(bits)) % k)).reshape(-1, k)


def modify_header(wire, field, value):
    values = list(HEADER.unpack_from(wire)); values[field] = value
    return HEADER.pack(*values) + wire[HEADER.size:]


def run_checks(output):
    output = Path(output).resolve(); artifacts = output.with_suffix('')
    assert not output.exists() and not artifacts.exists(), 'unique result paths required'
    output.parent.mkdir(parents=True, exist_ok=True); artifacts.mkdir()
    before = source_identities()
    record = dict(state='running', scope='receiver_framing_CPU_no_detector_AP', started_unix=time.time(),
                  protocol_sha256=sha(ROOT/'experiments/original-paper-scenarios/framing-protocol-001.md'),
                  sources_before=before, numpy=np.__version__, checks=[])
    fixtures = {}
    try:
        left, right = pair()
        for codec, parameter in [('jpeg', 50), ('jpeg2000', 4)]:
            wire, expected, _ = encode_pair(left, right, codec=codec, parameter=parameter)
            (artifacts/(codec+'.p6sb')).write_bytes(wire)
            # Construct two individually valid streams with DIFFERENT image heights.
            smaller, _, _ = encode_pair(left[:-1], right[:-1], codec=codec, parameter=parameter)
            fields = HEADER.unpack_from(wire); smallfields = HEADER.unpack_from(smaller)
            a = wire[HEADER.size:HEADER.size+fields[4]]
            b = smaller[HEADER.size+smallfields[4]:]
            mismatch = HEADER.pack(b'P6SB',1,fields[2],0,len(a),len(b),zlib.crc32(a+b))+a+b
            malformed = HEADER.pack(b'P6SB',1,fields[2],0,3,3,zlib.crc32(b'badbad'))+b'badbad'
            for k in (1296, 972):
                label = codec+'-'+str(k); original = blocks(wire,k)
                actual, images, item = receive_stereo(original,k=k)
                assert actual == wire and item['header_derived_bytes'] == len(wire)
                assert all(np.array_equal(a,b) for a,b in zip(images,expected))
                record['checks'].append(label+':actual_codec_roundtrip_without_byte_count')
                fixtures[label+'-valid'] = original
                mutations = [
                    ('magic',blocks(modify_header(wire,0,b'FAIL'),k),'header_magic'),
                    ('version',blocks(modify_header(wire,1,2),k),'header_version'),
                    ('codec',blocks(modify_header(wire,2,3),k),'header_codec'),
                    ('reserved',blocks(modify_header(wire,3,1),k),'header_reserved'),
                    ('zero_left',blocks(modify_header(wire,4,0),k),'header_zero_length'),
                    ('zero_right',blocks(modify_header(wire,5,0),k),'header_zero_length'),
                    ('overflow',blocks(modify_header(wire,4,0xffffffff),k),'header_capacity_overflow'),
                    ('CRC',blocks(modify_header(wire,6,fields[6]^1),k),'payload_CRC_failure'),
                    ('missing_block',original[:-1],'header_capacity_overflow'),
                    ('extra_block',np.vstack([original,np.zeros((1,k),dtype=np.uint8)]),'observed_block_count_mismatch'),
                    ('malformed_image',blocks(malformed,k),'image_decode_or_schema_failure'),
                    ('stereo_geometry',blocks(mismatch,k),'image_decode_or_schema_failure')]
                padding = original.copy(); assert original.size > 8*len(wire)
                padding.reshape(-1)[8*len(wire)] = 1
                mutations.append(('padding',padding,'nonzero_decoded_padding'))
                payload = original.copy(); payload.reshape(-1)[8*HEADER.size] ^= 1
                mutations.append(('payload',payload,'payload_CRC_failure'))
                nonbinary = original.astype(np.float64); nonbinary[0,0] = .5
                mutations.append(('nonbinary',nonbinary,'nonbinary_decoded_bits'))
                mutations.append(('width',original[:,:-1],'decoded_block_shape'))
                for name, fixture, reason in mutations:
                    got, images, result = receive_stereo(fixture,k=k)
                    assert got is None and images is None and result['state']=='erased'
                    assert result['erasure_reason']==reason, (label,name,result)
                    fixtures[label+'-'+name] = fixture
                    record['checks'].append(label+':erase_'+name)
        np.savez_compressed(artifacts/'decoded-fixtures.npz',**fixtures)
        assert source_identities() == before
        record.update(state='passed_receiver_only_framing_checks',sources_after=source_identities(),
                      artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in artifacts.iterdir()},
                      checks_count=len(record['checks']))
    except Exception as exc:
        record.update(state='failed',error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        record['finished_unix']=time.time()
        with output.open('x') as f: json.dump(record,f,indent=2)
    print(json.dumps({'state':record['state'],'checks':record['checks_count']}))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    run_checks(p.parse_args().output)
