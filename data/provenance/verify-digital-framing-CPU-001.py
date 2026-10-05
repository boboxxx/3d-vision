"""Independent raw-bit/header/codec/channel audit; no receiver/Sionna/Torch import."""
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import zlib

import numpy as np
from PIL import Image
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('historical_array_audit',ROOT/'data/provenance/verify-digital-transceiver-CPU-001.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def independent_reception(a,k):
    if a.ndim!=2 or not len(a) or a.shape[1]!=k:return 'decoded_block_shape',None,None
    if a.dtype.kind not in 'biuf' or not np.isfinite(a).all() or not np.isin(a,[0,1]).all():return 'nonbinary_decoded_bits',None,None
    bits=a.astype(np.uint8).ravel();head=np.packbits(bits[:160],bitorder='big').tobytes()
    if len(head)!=20:return 'truncated_header',None,None
    checks=[(head[:4]==b'P6SB','header_magic'),(head[4]==1,'header_version'),
            (head[5] in (1,2),'header_codec'),(int.from_bytes(head[6:8],'big')==0,'header_reserved')]
    for ok,reason in checks:
        if not ok:return reason,None,None
    nl=int.from_bytes(head[8:12],'big');nr=int.from_bytes(head[12:16],'big')
    if not nl or not nr:return 'header_zero_length',None,None
    size=20+nl+nr
    if size*8>bits.size:return 'header_capacity_overflow',None,None
    if math.ceil(size*8/k)!=len(a):return 'observed_block_count_mismatch',None,None
    if bits[size*8:].any():return 'nonzero_decoded_padding',None,None
    wire=np.packbits(bits[:size*8],bitorder='big').tobytes()
    if zlib.crc32(wire[20:])!=int.from_bytes(head[16:20],'big'):return 'payload_CRC_failure',None,None
    try:
        images=[];fmt='JPEG' if head[5]==1 else 'JPEG2000'
        for stream in (wire[20:20+nl],wire[20+nl:]):
            with Image.open(io.BytesIO(stream)) as im:
                if im.format!=fmt or im.mode!='RGB':raise ValueError('schema')
                images.append(np.array(im,dtype=np.uint8))
        if images[0].shape!=images[1].shape:raise ValueError('geometry')
    except Exception:return 'image_decode_or_schema_failure',None,None
    return None,wire,images


def audit_local_manifest(path):
    d=json.loads(path.read_text());assert d['state']=='passed_receiver_only_framing_checks'
    assert len(d['checks'])==d['checks_count']==68 and d['sources_before']==d['sources_after']
    for p,h in d['artifact_sha256'].items():assert sha(ROOT/p)==h
    with np.load(path.with_suffix('')/'decoded-fixtures.npz',allow_pickle=False) as f:
        assert len(f.files)==68
        reasons={'magic':'header_magic','version':'header_version','codec':'header_codec','reserved':'header_reserved',
                 'zero_left':'header_zero_length','zero_right':'header_zero_length','overflow':'header_capacity_overflow',
                 'CRC':'payload_CRC_failure','missing_block':'header_capacity_overflow','extra_block':'observed_block_count_mismatch',
                 'malformed_image':'image_decode_or_schema_failure','stereo_geometry':'image_decode_or_schema_failure',
                 'padding':'nonzero_decoded_padding','payload':'payload_CRC_failure','nonbinary':'nonbinary_decoded_bits','width':'decoded_block_shape'}
        for key in f.files:
            codec,k,name=key.split('-',2);reason,wire,images=independent_reception(f[key],int(k))
            if name=='valid':assert reason is None and wire==(path.with_suffix('')/(codec+'.p6sb')).read_bytes()
            else:assert reason==reasons[name] and wire is None and images is None
    return dict(path=str(path.relative_to(ROOT)),sha256=sha(path),independent_checks=68)


def main():
    path=ROOT/'data/engineering/artemis-digital-framing-CPU-001.json';d=json.loads(path.read_text())
    assert d['state']=='passed_actual_receiver_framing_audit_pending'
    assert len(d['packets'])==10 and d['sources_before']==d['sources_after'] and d['pip_freeze_before']==d['pip_freeze_after']
    launch=json.loads((ROOT/'data/engineering/artemis-digital-framing-CPU-001-launch.json').read_text())
    tpath=ROOT/'data/engineering/artemis-digital-framing-terminal-001.json';terminal=json.loads(tpath.read_text())
    assert terminal['state']=='completed_exit0_CPU_only' and terminal['job_id']==launch['job_id']==d['job_id']=='11424046'
    assert terminal['sacct_rows'] and all(r['State']=='COMPLETED' and r['ExitCode']=='0:0' for r in terminal['sacct_rows'])
    assert all('gres/gpu' not in r['AllocTRES'] for r in terminal['sacct_rows'])
    for p,h in terminal['fresh_file_sha256'].items():assert sha(ROOT/p)==h
    for p,h in launch['source_sha256'].items():assert sha(ROOT/p)==h
    for p,h in d['artifact_sha256'].items():assert sha(ROOT/p)==h and terminal['fresh_file_sha256'][p]==h
    assert d['protocol_sha256']==sha(ROOT/'experiments/original-paper-scenarios/framing-protocol-001.md')
    for p,h in d['sources_before'].items():
        if not p.startswith('sionna/'):assert sha(ROOT/p)==h
    historical=json.loads((ROOT/'data/engineering/artemis-digital-transceiver-CPU-001-launch.json').read_text())
    for p,h in historical['files_sha256'].items():assert sha(ROOT/p)==h
    local=[audit_local_manifest(ROOT/p) for p in ['data/engineering/digital-framing-local-CPU-002.json','data/engineering/artemis-digital-framing-local-CPU-001.json']]
    pixelpath=ROOT/'data/engineering/artemis-digital-framing-pixel-supplement-001.json'
    pixel=json.loads(pixelpath.read_text())
    assert pixel['state']=='passed_all6_independent_on_platform_pixel_redecodes'
    assert pixel['source_manifest_sha256']==sha(path) and len(pixel['views'])==6
    assert pixel['code_sha256']==sha(ROOT/'data/provenance/capture-framing-codec-pixels-001.py')
    assert pixel['protocol_sha256']==sha(ROOT/'experiments/original-paper-scenarios/framing-pixel-supplement-001.md')
    pixelterminalpath=ROOT/'data/engineering/artemis-digital-framing-pixel-terminal-001.json'
    pixelterminal=json.loads(pixelterminalpath.read_text())
    assert pixelterminal['state']=='completed_exit0_CPU_only' and pixelterminal['job_id']==pixel['job_id']=='11424047'
    assert pixelterminal['sacct_rows'] and all(r['State']=='COMPLETED' and r['ExitCode']=='0:0' for r in pixelterminal['sacct_rows'])
    assert all('gres/gpu' not in r['AllocTRES'] for r in pixelterminal['sacct_rows'])
    for p,h in pixelterminal['fresh_file_sha256'].items():assert sha(ROOT/p)==h
    pixelviews={v['packet']:v for v in pixel['views']}
    verified=[];pixel_differences=[]
    for item in d['packets']:
        r=item['diagnostic'];config=r['identity']['configuration']
        k,n,bps=(1296,1944,6) if config=='ldpc_2_3_qam64' else (972,1944,8)
        assert (r['identity']['k'],r['identity']['n'],r['identity']['bits_per_symbol'])==(k,n,bps)
        source=(ROOT/item['source_wire_path']).read_bytes()
        sourcebits=np.unpackbits(np.frombuffer(source,dtype=np.uint8),bitorder='big')
        blocks=math.ceil(len(sourcebits)/k);padded=np.pad(sourcebits,(0,blocks*k-len(sourcebits))).reshape(blocks,k)
        with np.load(ROOT/item['arrays_path'],allow_pickle=False) as f:a={key:f[key] for key in f.files}
        np.testing.assert_array_equal(a['source_padded'],padded)
        for key in ('coded','decoded'):old.array_check(a[key],r[key]);assert np.isin(a[key],[0,1]).all()
        assert a['coded'].shape==(blocks,n) and a['decoded'].shape==(blocks,k)
        labels=((np.arange(2**bps)[:,None]>>np.arange(bps-1,-1,-1))&1).astype(np.uint8)
        points=np.array([old.pam(b[0::2])+1j*old.pam(b[1::2]) for b in labels])/math.sqrt(2*(2**bps-1)/3)
        idx=a['coded'].reshape(-1,bps).astype(np.int64).dot(2**np.arange(bps-1,-1,-1));N=len(idx)
        np.testing.assert_allclose(a['transmitted'],points[idx],rtol=1e-15,atol=1e-15)
        for key in ('transmitted','received','noise','fading','effective_noise_variance'):old.array_check(a[key],r['channel'][key])
        kind=r['channel']['channel'];snr=r['channel']['SNR_dB'];n0=1e-6 if kind=='identity' else 10**(-snr/10)
        noisegen=np.random.Generator(np.random.PCG64(1901));fadegen=np.random.Generator(np.random.PCG64(1902))
        assert r['channel']['rng_before']==dict(noise=noisegen.bit_generator.state,fading=fadegen.bit_generator.state)
        noise=np.zeros(N,dtype=np.complex128);fade=np.ones(N,dtype=np.complex128)
        if kind!='identity':
            draw=noisegen.standard_normal((N,2))*math.sqrt(n0/2);noise=draw[:,0]+1j*draw[:,1]
            if kind=='rayleigh':
                draw=fadegen.standard_normal((N,2))/math.sqrt(2);fade=draw[:,0]+1j*draw[:,1]
        np.testing.assert_array_equal(a['noise'],noise);np.testing.assert_array_equal(a['fading'],fade)
        assert r['channel']['rng_after']==dict(noise=noisegen.bit_generator.state,fading=fadegen.bit_generator.state)
        expected=(fade*a['transmitted']+noise)/fade
        np.testing.assert_allclose(a['received'],expected,rtol=1e-14,atol=1e-14)
        np.testing.assert_allclose(a['effective_noise_variance'],n0/np.abs(fade)**2,rtol=3e-15,atol=0)
        actualLLR=a['llr'].reshape(N,bps);maxerror=0.
        for start in range(0,N,2048):
            stop=min(start+2048,N);scores=-np.abs(a['received'][start:stop,None]-points[None,:])**2/a['effective_noise_variance'][start:stop,None]
            expectedLLR=np.column_stack([logsumexp(scores[:,labels[:,b]==1],axis=1)-logsumexp(scores[:,labels[:,b]==0],axis=1) for b in range(bps)])
            np.testing.assert_allclose(actualLLR[start:stop],expectedLLR,rtol=5e-13,atol=1e-9)
            maxerror=max(maxerror,float(np.max(np.abs(actualLLR[start:stop]-expectedLLR))))
        reason,wire,images=independent_reception(a['decoded'],k)
        recv=item['reception'];assert recv['erasure_reason']==reason and recv['state']==('received' if reason is None else 'erased')
        assert recv['receiver_inputs']==['decoded_information_blocks','public_code_k'] and not recv['source_length_side_channel']
        assert item['physical_accounting_on_erasure_unchanged'] and item['attempted_uses']==r['channel']['uses']==N
        np.testing.assert_allclose(item['attempted_energy'],np.sum(np.abs(a['transmitted'])**2),rtol=1e-15,atol=1e-10)
        assert r['source_bits']==len(sourcebits) and r['source_bytes']==len(source) and r['blocks']==blocks
        assert r['source_padding_bits']==blocks*k-len(sourcebits) and r['coded_bits']==blocks*n and r['QAM_padding_bits']==0
        truncated=a['decoded'].ravel()[:len(sourcebits)]
        assert r['source_bit_errors']==int(np.count_nonzero(truncated!=sourcebits))
        assert r['source_block_errors']==int(np.count_nonzero(np.any(a['decoded']!=padded,axis=1)))
        recovered=np.packbits(truncated,bitorder='big').tobytes()
        assert r['exact_payload_recovered']==(recovered==source)
        assert r['source_padding_valid']==bool(not np.any(a['decoded'].ravel()[len(sourcebits):]))
        assert r['received_sha256']==hashlib.sha256(recovered).hexdigest()
        target=ROOT/'data/engineering/artemis-digital-framing-CPU-001'/(item['name']+'.p6sb')
        if reason is None:
            assert wire==target.read_bytes() and recv['header_derived_bytes']==len(wire)
            assert recv['native_shape']==list(images[0].shape)
            supplementary=pixelviews[item['name']]
            assert supplementary['wire_sha256']==sha(target) and supplementary['fixture_sha256']==sha(ROOT/item['arrays_path'])
            pixelarrays=ROOT/supplementary['pixel_arrays_path']
            assert supplementary['pixel_arrays_sha256']==sha(pixelarrays)
            with np.load(pixelarrays,allow_pickle=False) as f:serverimages=[f['left'],f['right']]
            serverhashes=[hashlib.sha256(np.ascontiguousarray(im).tobytes()).hexdigest() for im in serverimages]
            assert supplementary['image_sha256']==recv['received_image_sha256']==serverhashes
            views=[]
            for localimage,serverimage in zip(images,serverimages):
                assert localimage.shape==serverimage.shape and serverimage.dtype==np.uint8
                delta=np.abs(localimage.astype(np.int16)-serverimage.astype(np.int16))
                views.append(dict(exact_pixel_equality=bool(not delta.any()),changed_values=int(np.count_nonzero(delta)),
                                  total_values=int(delta.size),maximum_absolute_difference=int(delta.max()),
                                  mean_absolute_difference=float(delta.mean())))
            pixel_differences.append(dict(packet=item['name'],scope='exploratory_cross_platform_decode_no_acceptance_bound',views=views))
        else:assert not target.exists() and wire is None and images is None
        if kind=='identity':assert reason is None and wire==source and len(wire)==90545 and images[0].shape==(370,1224,3)
        verified.append(dict(name=item['name'],erasure_reason=reason,source_bit_errors=r['source_bit_errors'],attempted_uses=N,
                             attempted_energy=item['attempted_energy'],APP_max_roundoff=maxerror))
    assert d['successful_packets']==sum(p['erasure_reason'] is None for p in verified)
    assert d['erased_packets']==sum(p['erasure_reason'] is not None for p in verified)
    out=ROOT/'data/provenance/digital-framing-CPU-001-local-verification.json'
    result=dict(state='passed_all10_actual_receiver_header_packet_audits',scope='CPU_engineering_ideal_sync_public_configuration_no_AP',
                manifest_sha256=sha(path),terminal_sha256=sha(tpath),artifact_count=len(d['artifact_sha256']),
                local_receiver_checks=local,verified=verified,pixel_supplement_sha256=sha(pixelpath),
                pixel_terminal_sha256=sha(pixelterminalpath),cross_platform_pixel_differences=pixel_differences,
                cross_platform_native_pixel_equality=False)
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:result[k] for k in ('state','artifact_count')}))


if __name__=='__main__':main()
