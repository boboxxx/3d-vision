"""Byte identity with pinned independent C++ rANS and adversarial framing checks."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import time
import zlib

import numpy as np
import codec as c

ROOT = Path(__file__).resolve().parents[3]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); assert not args.output.exists()
    cdfs, identity = c.tables(); checks=[]; rejected=[]; fixtures=[]
    reference=ROOT/'data/provenance/ecsic-rans-reference-001'
    provenance=json.loads((reference/'source.json').read_text())
    assert sha(reference/'rans_byte.h') == provenance['sha256']
    assert identity == '507d22e35fb0bb605f05a3312600a8aff3ff10f1ec1c4964358cb390e35d56e5'
    checks.append('full_positive_CDF_support_identity')
    compiler=shutil.which('clang++') or shutil.which('g++'); assert compiler
    rng=np.random.Generator(np.random.PCG64(1908))
    cases=[('full_alphabet_cross_tables',np.tile(np.arange(512),256),np.repeat(np.arange(256),512)),
           ('zeros',np.full(10000,255),np.zeros(10000,dtype=int)),
           ('random_varying_tables',rng.integers(0,512,8192),rng.integers(0,256,8192))]
    with tempfile.TemporaryDirectory(prefix='paper6-rans-') as temporary:
        t=Path(temporary); binary=t/'reference'
        compile_result=subprocess.run([compiler,'-std=c++11','-O2','-I',str(reference),str(c.HERE/'reference.cpp'),'-o',str(binary)],capture_output=True,text=True)
        assert compile_result.returncode==0,compile_result.stderr
        for name,symbols,indices in cases:
            fixture=t/(name+'.bin'); target=t/(name+'.rans')
            fixture.write_bytes(np.stack([symbols,indices],axis=1).astype('<u4').tobytes())
            subprocess.run([str(binary),str(c.HERE/'cdf-u32le.bin'),str(fixture),str(target)],check=True)
            actual=c.rans_encode(symbols,indices,cdfs)
            assert actual==target.read_bytes() and np.array_equal(c.rans_decode(actual,indices,cdfs),symbols)
            fixtures.append(dict(name=name,symbols=len(symbols),bytes=len(actual),sha256=hashlib.sha256(actual).hexdigest(),independent_reference_sha256=sha(target)))
        compiler_version=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0]
    checks.append('C++_reference_exact_bytes_and_independent_roundtrip')
    values=np.asarray([-2**24+1,-65536,-256,-255,-1,0,1,255,256,65536,2**24-1],dtype=np.int32)
    values=np.concatenate([values,rng.integers(-100000,100001,4096,dtype=np.int32)])
    scales=np.exp(rng.uniform(np.log(.01),np.log(1000),len(values)))
    rb,eb=c.encode_stream(values,scales,cdfs)
    assert np.array_equal(c.decode_stream(rb,eb,scales,values.shape,cdfs),values)
    checks.append('all_escape_boundaries_extremes_random_exact')
    ids=c.scale_ids(np.array([-.1,0,.1,256.,1000.]),(5,))
    assert ids.tolist()==[0,0,0,255,255]
    centers=np.exp(np.linspace(np.log(.1),np.log(256.),256))
    assert np.array_equal(c.scale_ids(centers,(256,)),np.arange(256))
    checks.append('scale_index_centers_and_clipping')
    streams={}
    for i,name in enumerate(c.ORDER):
        shape=c.shape_of(i,[32,64]); a=np.zeros(shape,dtype=np.int32)
        streams[name]=c.encode_stream(a,1.,cdfs)
    blob=c.pack([31,63],[32,64],streams,identity)
    decoded=c.unpack(blob,identity)
    assert decoded['original_hw']==[31,63] and decoded['padded_hw']==[32,64]
    assert decoded['streams']==streams and len(blob)==166+sum(len(a)+len(b) for a,b in streams.values())
    for i,name in enumerate(c.ORDER):
        assert not c.decode_stream(*decoded['streams'][name],1.,c.shape_of(i,[32,64]),cdfs).any()
    checks.append('four_stream_container_full_overhead_and_roundtrip')
    def reject(name,call):
        try: call()
        except ValueError: rejected.append(name)
        else: raise AssertionError('accepted malformed input: '+name)
    reject('empty_rans',lambda:c.rans_decode(b'',np.array([0]),cdfs))
    reject('invalid_initial_state',lambda:c.rans_decode(b'\0'*4,np.array([0]),cdfs))
    reject('empty_symbols',lambda:c.rans_encode([],[],cdfs))
    reject('wrong_model_length',lambda:c.rans_encode([0],[0,0],cdfs))
    reject('truncated_rans',lambda:c.rans_decode(rb[:-1],c.scale_ids(scales,values.shape),cdfs))
    reject('extra_rans_bytes',lambda:c.rans_decode(rb+b'\0',c.scale_ids(scales,values.shape),cdfs))
    reject('unused_escape',lambda:c.decode_stream(rb,eb+b'\0',scales,values.shape,cdfs))
    reject('truncated_escape',lambda:c.decode_stream(rb,eb[:-1],scales,values.shape,cdfs))
    reject('overlong_escape',lambda:c.read_varint(b'\x80\x84\x00',0))
    reject('within_support_escape',lambda:c.read_varint(b'\x02',0))
    reject('unbounded_escape',lambda:c.read_varint(b'\xff'*5,0))
    reject('nonfinite_scale',lambda:c.scale_ids(np.nan,(1,)))
    reject('float_residual',lambda:c.encode_stream(values.astype(float),scales,cdfs))
    reject('residual_range',lambda:c.encode_stream(np.array([2**24],dtype=np.int32),1.,cdfs))
    reject('missing_stream',lambda:c.pack([31,63],[32,64],dict(list(streams.items())[:3]),identity))
    reject('wrong_order',lambda:c.pack([31,63],[32,64],dict(reversed(list(streams.items()))),identity))
    reject('invalid_dimensions',lambda:c.pack([0,63],[32,64],streams,identity))
    reject('oversized_dimensions',lambda:c.pack([2050,63],[2080,64],streams,identity))
    reject('invalid_padding',lambda:c.pack([1,63],[64,64],streams,identity))
    for pos in list(range(162))+[162,len(blob)//2,len(blob)-5,len(blob)-4,len(blob)-1]:
        bad=bytearray(blob);bad[pos]^=1
        reject('bit_flip_'+str(pos),lambda bad=bad:c.unpack(bytes(bad),identity))
    for n in (0,1,4,165,181,len(blob)-1):
        reject('truncation_'+str(n),lambda n=n:c.unpack(blob[:n],identity))
    reject('appended_byte',lambda:c.unpack(blob+b'\0',identity))
    def mutate_header(position,data):
        bad=bytearray(blob[:-4]);bad[position:position+len(data)]=data
        return bytes(bad)+struct.pack('>I',zlib.crc32(bad))
    for name,position,data in [('magic',0,b'FAIL'),('version',4,b'\x02'),('count',5,b'\x03'),
                                ('zero_height',6,b'\0\0'),('weights',14,b'\0'*32),('config',46,b'\0'*32),('CDF',78,b'\0'*32),
                                ('stream_id',110,b'\x03'),('symbol_count',111,b'\0'*4),('rans_length',115,b'\0'*4),
                                ('escape_length',119,b'\xff'*4)]:
        bad=mutate_header(position,data)
        reject('valid_CRC_bad_'+name,lambda bad=bad:c.unpack(bad,identity))
    checks.append('malformed_bytes_and_CRC_consistent_bad_headers_rejected')
    output=dict(state='passed',checked_unix=time.time(),checks=checks,rejected_cases=rejected,
                CDF_sha256=identity,reference=provenance,compiler_version=compiler_version,
                reference_fixtures=fixtures,source_sha256={p.name:sha(p) for p in c.HERE.iterdir() if p.is_file()},
                scope='entropy CPU engineering fixtures only; no ECSIC native image or AP measurement')
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(state='passed',checks=len(checks),rejections=len(rejected),reference_fixtures=fixtures)))


if __name__=='__main__':main()
