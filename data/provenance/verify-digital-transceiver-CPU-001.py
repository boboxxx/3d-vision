"""Independent transferred-array audit; no transceiver/Sionna/Torch imports."""
from pathlib import Path
import hashlib,json,math
import numpy as np
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[2]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def array_check(a,d):
    assert list(a.shape)==d['shape'] and a.dtype.str==d['dtype']
    assert hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()==d['sha256']

def pam(bits):
    return int(1-2*int(bits[0]))*(2**(len(bits)-1)-pam(bits[1:])) if len(bits)>1 else 1-2*int(bits[0])


def main():
    path=ROOT/'data/engineering/artemis-digital-transceiver-CPU-001.json'
    d=json.loads(path.read_text());assert d['state']=='passed_actual_LDPC_QAM_CPU_checks'
    terminal=json.loads((ROOT/'data/engineering/artemis-digital-terminal-001.json').read_text())
    assert terminal['state']=='completed_exit0_CPU_only'
    assert len(d['checks'])==12 and len(d['packets'])==8 and d['sources_before']==d['sources_after']
    for p,v in d['artifact_sha256'].items():assert sha(ROOT/p)==v
    verified=[]
    for r in d['packets']:
        config=r['identity']['configuration'];k,n,bps=(1296,1944,6) if config=='ldpc_2_3_qam64' else (972,1944,8)
        assert (r['identity']['k'],r['identity']['n'],r['identity']['bits_per_symbol'])==(k,n,bps)
        kind=r['channel']['channel'];native=r['packet']=='native000000_JP2rates30'
        basename=config+('-native-identity' if native else '-'+kind)
        base=ROOT/'data/engineering/artemis-digital-transceiver-CPU-001'/basename
        with np.load(str(base)+'.npz',allow_pickle=False) as f:a={key:f[key] for key in f.files}
        source=(ROOT/'data/engineering/original-scenarios-CPU-smoke-001/jpeg2000-30.p6sb').read_bytes() if native else bytes(range(256))+b'P6LDPC'
        received=Path(str(base)+('.p6sb' if native else '.bin')).read_bytes()
        bits=np.unpackbits(np.frombuffer(source,dtype=np.uint8),bitorder='big')
        blocks=math.ceil(len(bits)/k);padded=np.pad(bits,(0,blocks*k-len(bits))).reshape(blocks,k)
        np.testing.assert_array_equal(a['source_padded'],padded)
        assert r['source_bytes']==len(source) and r['source_bits']==len(bits) and r['blocks']==blocks
        assert r['coded_bits']==blocks*n and r['source_padding_bits']==blocks*k-len(bits) and r['QAM_padding_bits']==0
        assert a['coded'].shape==(blocks,n) and a['decoded'].shape==(blocks,k)
        for key in ['coded','decoded']:array_check(a[key],r[key]);assert np.isin(a[key],[0,1]).all()
        labels=((np.arange(2**bps)[:,None] >> np.arange(bps-1,-1,-1)) & 1).astype(np.uint8)
        points=np.array([pam(b[0::2])+1j*pam(b[1::2]) for b in labels],dtype=np.complex128)/math.sqrt(2*(2**bps-1)/3)
        # Reconstruct actual bit labeling independently from the NR PAM expression.
        idx=a['coded'].reshape(-1,bps).astype(np.int64).dot(2**np.arange(bps-1,-1,-1))
        np.testing.assert_allclose(a['transmitted'],points[idx],rtol=1e-15,atol=1e-15)
        for key in ['transmitted','received','noise','fading','effective_noise_variance']:array_check(a[key],r['channel'][key])
        n0=1e-6 if kind=='identity' else 10**(-10/10)
        N=len(idx);rngs={name:np.random.Generator(np.random.PCG64()) for name in ['noise','fading']}
        for name in rngs:rngs[name].bit_generator.state=r['channel']['rng_before'][name]
        noise=np.zeros(N,dtype=np.complex128);fade=np.ones(N,dtype=np.complex128)
        if kind!='identity':
            draws=rngs['noise'].standard_normal((N,2))*math.sqrt(n0/2);noise=draws[:,0]+1j*draws[:,1]
            if kind=='rayleigh':
                draws=rngs['fading'].standard_normal((N,2))/math.sqrt(2);fade=draws[:,0]+1j*draws[:,1]
        np.testing.assert_array_equal(a['noise'],noise);np.testing.assert_array_equal(a['fading'],fade)
        for name in rngs:assert rngs[name].bit_generator.state==r['channel']['rng_after'][name]
        expected=(fade*a['transmitted']+noise)/fade
        maximum=float(np.max(np.abs(a['received']-expected)))
        tolerance=64*np.finfo(np.float64).eps*max(1,float(np.max(np.abs(expected))))
        assert maximum<=tolerance
        np.testing.assert_allclose(a['effective_noise_variance'],n0/np.abs(fade)**2,rtol=3e-15,atol=0)
        # Validate every actual demapper output; no use of the implementation's function.
        llr=a['llr'].reshape(N,bps);max_llr_error=0.
        for start in range(0,N,2048):
            end=min(start+2048,N)
            scores=-np.abs(a['received'][start:end,None]-points[None,:])**2/a['effective_noise_variance'][start:end,None]
            expected_llr=np.column_stack([logsumexp(scores[:,labels[:,b]==1],axis=1)-logsumexp(scores[:,labels[:,b]==0],axis=1) for b in range(bps)])
            max_llr_error=max(max_llr_error,float(np.max(np.abs(expected_llr-llr[start:end]))))
            np.testing.assert_allclose(llr[start:end],expected_llr,rtol=5e-13,atol=1e-9)
        decoded=a['decoded'].reshape(-1)[:len(bits)]
        assert np.packbits(decoded,bitorder='big').tobytes()==received
        assert r['source_bit_errors']==int(np.count_nonzero(decoded!=bits))
        assert r['source_block_errors']==int(np.count_nonzero(np.any(a['decoded']!=padded,axis=1)))
        assert r['exact_payload_recovered']==(received==source)
        assert r['source_padding_valid']==bool(not np.any(a['decoded'].reshape(-1)[len(bits):]))
        assert r['source_sha256']==hashlib.sha256(source).hexdigest() and r['received_sha256']==hashlib.sha256(received).hexdigest()
        assert r['channel']['uses']==blocks*n//bps and r['channel']['pilot_uses']==0
        np.testing.assert_allclose(r['channel']['actual_energy'],np.sum(np.abs(a['transmitted'])**2),rtol=1e-15,atol=1e-10)
        if kind=='identity':assert received==source and r['source_padding_valid']
        verified.append(dict(configuration=config,packet=r['packet'],channel=kind,source_bit_errors=r['source_bit_errors'],actual_energy=r['channel']['actual_energy'],uses=N,channel_max_roundoff=maximum,channel_tolerance=tolerance,APP_max_roundoff=max_llr_error))
    result=dict(state='passed_all8_transferred_actual_packet_audits',scope='CPU_engineering_not_main_noise_AP_or_exact_original_configuration',manifest_sha256=sha(path),terminal_sha256=sha(ROOT/'data/engineering/artemis-digital-terminal-001.json'),artifacts=16,verified=verified)
    out=ROOT/'data/provenance/digital-transceiver-CPU-001-local-verification.json'
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(result['state'])


if __name__=='__main__':main()
