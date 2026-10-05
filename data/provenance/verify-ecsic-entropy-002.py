"""Fresh complete P6EC parser, independent compiled rANS and full array audit."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import time
import zlib

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
RUN=Path('/mnt/d/paper6/runs/ecsic-entropy-stageB-002')
PREVIOUS=Path('/mnt/d/paper6/runs/ecsic-recovery-stageA-001')
ORDER=('z_left','z_right','y_left','y_right')


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(p.read_text())


def describe(a):
    return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest(),
                finite=bool(np.isfinite(a).all()),min=float(a.min()),max=float(a.max()))


def escapes(values):
    out=bytearray()
    for value in values:
        value=int(value);u=value*2 if value>=0 else -(value+1)*2+1
        while True:
            low=u%128;u//=128;out.append(low+(128 if u else 0))
            if not u:break
    return bytes(out)


def main():
    output=ROOT/'data/provenance/ecsic-entropy-stageB-002-audit.json';assert not output.exists()
    run=read(RUN/'manifest.json');assert run['state']=='passed' and run['actual_entropy_source_bytes']
    try:os.kill(run['pid'],0)
    except ProcessLookupError:pass
    else:raise RuntimeError('native PID still live')
    assert not run['physical_channel_executed'] and not run['KITTI_AP_measured']
    assert all(sha(ROOT/k)==v for k,v in run['sources_sha256'].items())
    assert run['protocol_sha256']==sha(ROOT/'experiments/original-paper-scenarios/ecsic-entropy-protocol-001.md')
    aa=read(ROOT/'data/provenance/ecsic-recovery-stageA-001-audit.json');stage_a=read(PREVIOUS/'manifest.json')
    assert aa['state']=='passed' and sha(PREVIOUS/'manifest.json')==aa['manifest_sha256']
    assert run['stage_A_audit_sha256']==sha(ROOT/'data/provenance/ecsic-recovery-stageA-001-audit.json')
    code=ROOT/'experiments/original-paper-scenarios/ecsic-entropy-code'
    for host in ('local','sheng'):
        regression=read(ROOT/'data/engineering'/f'ecsic-output-repair-{host}-001.json')
        assert regression['state']=='passed' and regression['blocked_reads']==1
        assert regression['native_source_sha256']==sha(code/'native.py')
    cdf_bytes=(code/'cdf-u32le.bin').read_bytes()
    assert hashlib.sha256(cdf_bytes).hexdigest()==run['CDF_sha256']=='507d22e35fb0bb605f05a3312600a8aff3ff10f1ec1c4964358cb390e35d56e5'
    cdf=np.frombuffer(cdf_bytes,dtype='<u4').reshape(256,513)
    refdir=ROOT/'data/provenance/ecsic-rans-reference-001';refinfo=read(refdir/'source.json')
    assert sha(refdir/'rans_byte.h')==refinfo['sha256']
    results={}
    with tempfile.TemporaryDirectory(prefix='ecsic-entropy-audit-') as temp:
        t=Path(temp);binary=t/'reference';compiler=shutil.which('g++') or shutil.which('clang++');assert compiler
        subprocess.run([compiler,'-std=c++11','-O2','-I',str(refdir),str(code/'reference.cpp'),'-o',str(binary)],check=True)
        for case in ('synthetic32x64','training000000'):
            d=RUN/case;previous=PREVIOUS/case;item=run['cases'][case];rec=read(d/'receiver.json')
            assert all(sha(d/k)==v for k,v in item['files'].items())
            assert all(sha(previous/k)==v for k,v in stage_a['cases'][case]['files'].items())
            assert rec['state']=='passed' and rec['source_NPZ_read_barrier'] and rec['no_parameter_gradients']
            assert rec['full_states_before']==rec['full_states_after']==stage_a['full_states_after']
            assert rec['calls']==dict(E=0,HE=0,HD=1,D=1) and rec['sources_sha256']==run['sources_sha256']
            assert rec['source_sha256']==stage_a['source_sha256']
            blob=(d/'source.p6ec').read_bytes();assert len(blob)<=32*1024*1024
            assert hashlib.sha256(blob).hexdigest()==rec['container_sha256'] and len(blob)==item['container_bytes']==rec['container_bytes']
            assert zlib.crc32(blob[:-4])==int.from_bytes(blob[-4:],'big')
            magic,ver,count,h,w,ph,pw,model,config,table=struct.unpack('>4sBB4H32s32s32s',blob[:110])
            assert (magic,ver,count)==(b'P6EC',1,4)
            assert model.hex()==stage_a['model_sha256'] and config.hex()==stage_a['config_sha256'] and table.hex()==run['CDF_sha256']
            assert 0<h<=ph<=2048 and 0<w<=pw<=4096 and ph%32==pw%32==0 and ph-h<32 and pw-w<32
            assert [h,w]==item['original_hw'] and [ph,pw]==item['padded_hw']
            assert rec['output_sha256']==sha(d/'reconstructed.npz')
            cursor=162;total_ideal=0.;stream_results={}
            with np.load(d/'reconstructed.npz',allow_pickle=False) as dec, np.load(previous/'reference.npz',allow_pickle=False) as ref, np.load(previous/'symbols.npz',allow_pickle=False) as sy:
                assert set(dec.files)==set(ref.files)==set(item['exact_arrays']) and len(dec.files)==18
                assert {k:describe(dec[k]) for k in dec.files}==rec['arrays']
                for key in dec.files:
                    assert item['exact_arrays'][key] and np.isfinite(dec[key]).all() and np.array_equal(dec[key],ref[key])
                for i,name in enumerate(ORDER):
                    sid,n,nr,ne=struct.unpack('>BIII',blob[110+13*i:123+13*i]);factor=32 if i<2 else 8
                    shape=(1,48,ph//factor,pw//factor)
                    assert sid==i and n==np.prod(shape) and nr>=4 and ne<=4*n
                    rans=blob[cursor:cursor+nr];tail=blob[cursor+nr:cursor+nr+ne];cursor+=nr+ne
                    q=dec[name+'_symbols'];assert q.dtype==np.int32 and q.shape==shape
                    assert np.array_equal(q,sy[name]) and np.max(np.abs(q.astype(np.int64)))<2**24
                    assert np.array_equal(q.astype(np.float32)+dec[name+'_loc'],dec[name+'_hat'])
                    scales=np.broadcast_to(dec[name+'_scale'].astype(np.float64),shape)
                    ids=np.clip(np.rint((np.log(np.clip(scales,.1,256.))-np.log(.1))*(255/np.log(2560.))),0,255).astype(np.uint8).ravel()
                    assert describe(ids)==rec['derived_scale_indices'][name]==item['streams'][name]['derived_scale_indices']
                    values=q.ravel();mask=(values < -255)|(values > 255);codes=np.where(mask,511,values+255)
                    assert tail==escapes(values[mask])
                    fixture=t/(name+'.bin');expected=t/(name+'.rans')
                    fixture.write_bytes(np.stack([codes,ids],axis=1).astype('<u4').tobytes())
                    subprocess.run([str(binary),str(code/'cdf-u32le.bin'),str(fixture),str(expected)],check=True)
                    assert rans==expected.read_bytes()
                    freq=(cdf[ids.astype(int),codes+1]-cdf[ids.astype(int),codes]).astype(np.float64)
                    ideal=float(-np.log2(freq/65536).sum());total_ideal+=ideal+8*ne
                    details=item['streams'][name]
                    assert (details['symbols'],details['rans_bytes'],details['escape_bytes'],details['escape_symbols'])==(n,nr,ne,int(mask.sum()))
                    assert ideal==details['finite_CDF_ideal_bits']
                    stream_results[name]=dict(symbols=n,rans_bytes=nr,escape_bytes=ne,independent_Cxx_sha256=sha(expected))
            assert cursor==len(blob)-4 and item['header_CRC_bytes']==166
            assert item['container_bits']==8*len(blob)
            assert abs(total_ideal-item['estimated_CDF_plus_literal_bits'])<1e-8
            assert abs(item['actual_minus_estimate_bits']-(8*len(blob)-total_ideal))<1e-8
            results[case]=dict(container_bytes=len(blob),payload_sha256=sha(d/'source.p6ec'),exact_arrays=18,streams=stream_results)
    result=dict(state='passed',actual_native_terminal=True,native_pid=run['pid'],audited_unix=time.time(),
                manifest_sha256=sha(RUN/'manifest.json'),verifier_sha256=sha(Path(__file__)),cases=results,
                independent_reference=refinfo,complete_states=225,actual_entropy_bytes=True,physical_channel_or_AP_claim=False)
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(state='passed',cases=results)))


if __name__=='__main__':main()
