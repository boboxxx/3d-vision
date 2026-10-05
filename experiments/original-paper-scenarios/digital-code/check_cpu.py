"""Actual code/constellation/packet checks; no KITTI labels or detector."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import scipy
import torch
import sionna

from transceiver import DigitalTransceiver, app_demapper, array_record, labels

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/original-paper-scenarios/code'))
from image_codecs import decode_pair


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def sources():
    dirs={'digital':Path(__file__).parent,'scenario':ROOT/'experiments/original-paper-scenarios/code','sionna':Path(sionna.__file__).parent}
    return {prefix+'/'+str(p.relative_to(d)):sha(p) for prefix,d in dirs.items() for p in sorted(d.rglob('*'))
            if p.is_file() and p.suffix in ('.py','.csv') and '__pycache__' not in p.parts}


def main():
    output=ROOT/'data/engineering/artemis-digital-transceiver-CPU-001.json'
    artifacts=ROOT/'data/engineering/artemis-digital-transceiver-CPU-001'
    if output.exists() or artifacts.exists() or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('unique engineering and CPU Slurm allocation required')
    runtime=json.loads((ROOT/'data/engineering/artemis-digital-runtime-001.json').read_text())
    assert runtime['state']=='passed_independent_CPU_PHY_imports_sources_exact_RT_omitted'
    assert torch.__version__=='2.9.1+cpu' and sionna.__version__=='2.1.0'
    installed=Path(sionna.__file__).parent
    assert all(sha(installed/p)==v for p,v in runtime['installed_source_sha256'].items())
    torch.set_num_threads(4)
    before=sources(); artifacts.mkdir()
    record=dict(state='running',scope='actual_digital_CPU_engineering_ONLY_no_AP',job_id=os.environ['SLURM_JOB_ID'],
                started_at_unix=time.time(),protocol_sha256=sha(ROOT/'experiments/original-paper-scenarios/digital-protocol-001.md'),
                runtime_sha256=sha(ROOT/'data/engineering/artemis-digital-runtime-001.json'),
                sources_before=before,torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,sionna=sionna.__version__,
                checks=[],packets=[])
    with output.open('x') as f:json.dump(record,f,indent=2)
    try:
        payload=bytes(range(256))+b'P6LDPC'
        native=ROOT/'assets/digital-native-001/jpeg2000-30.p6sb'
        assert sha(native)=='a9e92e1c66e138f35497984e763db1b63b58be54878d5413b08d68571d3912cd'
        native_pair=decode_pair(native.read_bytes())
        assert native_pair[0].shape==native_pair[1].shape==(370,1224,3)
        for config in ('ldpc_2_3_qam64','ldpc_1_2_qam256'):
            link=DigitalTransceiver(config)
            # Check every point's unique label and unit ensemble power.
            assert len(np.unique(link.points))==2**link.bps
            assert abs(np.mean(np.abs(link.points)**2)-1)<1e-14
            llr=app_demapper(link.points,1e-6,link.points,link.bit_labels)
            np.testing.assert_array_equal(llr>0,link.bit_labels.astype(bool))
            # Independent adjacency assertion: nearest horizontal/vertical labels differ in one bit.
            for i,p in enumerate(link.points):
                delta=np.abs(link.points-p);nearest=np.flatnonzero(np.isclose(delta,np.min(delta[delta>0]),rtol=1e-12,atol=1e-12))
                assert np.all(np.sum(link.bit_labels[nearest]!=link.bit_labels[i],axis=1)==1)
            # Check APP sign and per-symbol N0 using independent direct likelihood sums.
            observations=link.points[[1,3,5]]+np.array([.013+.007j,-.009+.003j,.004-.011j])
            variances=np.array([.1,.3,.7]);actual=app_demapper(observations,variances,link.points,link.bit_labels)
            expected=np.empty_like(actual)
            for i,x in enumerate(observations):
                likelihood=np.exp(-np.abs(x-link.points)**2/variances[i])
                for b in range(link.bps):expected[i,b]=np.log(likelihood[link.bit_labels[:,b]==1].sum()/likelihood[link.bit_labels[:,b]==0].sum())
            np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
            record['checks'].append(config+':all_constellation_Gray_energy_APP_direct_likelihood')
            # Independent GF2 PCM multiplication, not encode-then-decode agreement alone.
            source=np.stack([np.zeros(link.k),np.ones(link.k),np.random.default_rng(11).integers(0,2,link.k)]).astype(np.float64)
            filled=np.pad(source,((0,0),(0,link.encoder.k_ldpc-link.k)))
            with torch.no_grad():mother=link.encoder._encode_fast(torch.from_numpy(filled)).cpu().numpy().reshape(3,link.encoder.n_ldpc).astype(np.int64)
            syndrome=np.asarray(link.encoder.pcm.astype(np.int64).dot(mother.T))%2
            assert not syndrome.any();np.testing.assert_array_equal(mother[:,:link.encoder.k_ldpc],filled)
            without_filler=np.concatenate([mother[:,:link.k],mother[:,link.encoder.k_ldpc:]],axis=1)
            independently_matched=without_filler[:,2*link.encoder.z:][:,:link.n][:,link.encoder.out_int.cpu().numpy()]
            with torch.no_grad():coded=link.encoder(torch.from_numpy(source)).cpu().numpy()
            np.testing.assert_array_equal(coded,independently_matched)
            record['checks'].append(config+':zero_syndrome_systematic_filler_RV0_interleaver')
            for kind in ('identity','awgn','rayleigh'):
                np_state=copy.deepcopy(np.random.get_state())
                n=np.random.Generator(np.random.PCG64(1901));h=np.random.Generator(np.random.PCG64(1902))
                received,item,arrays=link.transmit(payload,kind=kind,snr_db=10,noise_rng=n,fading_rng=h)
                after=np.random.get_state()
                assert all(np.array_equal(a,b) for a,b in zip(np_state,after))
                repeated,item2,arrays2=link.transmit(payload,kind=kind,snr_db=10,
                    noise_rng=np.random.Generator(np.random.PCG64(1901)),fading_rng=np.random.Generator(np.random.PCG64(1902)))
                assert repeated==received and item2==item
                assert all(np.array_equal(arrays[k],arrays2[k]) for k in arrays)
                assert item['coded_bits']==item['blocks']*1944
                assert item['source_bits']==2096 and item['source_bytes']==262
                assert item['blocks']==(2 if link.k==1296 else 3)
                assert item['source_padding_bits']==(496 if link.k==1296 else 820)
                assert item['channel']['uses']==(648 if link.k==1296 else 729)
                assert item['QAM_padding_bits']==0
                if kind=='identity':
                    assert received==payload and item['source_padding_valid']
                    assert item['channel']['rng_before']==item['channel']['rng_after']
                # This reconstructs the channel from retained draws, including exact CSI variance.
                np.testing.assert_array_equal(arrays['received'],(arrays['fading']*arrays['transmitted']+arrays['noise'])/arrays['fading'])
                expected_n0=1e-6 if kind=='identity' else .1
                np.testing.assert_allclose(arrays['effective_noise_variance'],expected_n0/np.abs(arrays['fading'])**2,rtol=2e-15,atol=0)
                name=config+'-'+kind
                np.savez_compressed(artifacts/(name+'.npz'),**arrays)
                (artifacts/(name+'.bin')).write_bytes(received)
                item['packet']='synthetic262bytes';record['packets'].append(item)
                record['checks'].append(name+':actual_chain_padding_counts_draw_replay')
            # Full real source bitstream through actual code/QAM/BP, no image fallback.
            received,item,arrays=link.transmit(native.read_bytes(),kind='identity',snr_db=10,
                    noise_rng=np.random.Generator(np.random.PCG64(1901)),fading_rng=np.random.Generator(np.random.PCG64(1902)))
            assert received==native.read_bytes() and item['source_padding_valid']
            decoded_pair=decode_pair(received)
            assert all(np.array_equal(a,b) for a,b in zip(native_pair,decoded_pair))
            name=config+'-native-identity';(artifacts/(name+'.p6sb')).write_bytes(received)
            np.savez_compressed(artifacts/(name+'.npz'),**arrays)
            item['packet']='native000000_JP2rates30';record['packets'].append(item)
            record['checks'].append(name+':complete_bytes_CRC_native370x1224')
        after=sources();assert before==after
        record.update(state='passed_actual_LDPC_QAM_CPU_checks',sources_after=after,
                      artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(artifacts.iterdir())},
                      native_stream_sha256=sha(native),native_shape=list(native_pair[0].shape))
    except Exception as e:
        record.update(state='failed',error=f'{type(e).__name__}: {e}')
        raise
    finally:
        record['finished_at_unix']=time.time();output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(state=record['state'],checks=len(record['checks']),packets=len(record['packets']))))


if __name__=='__main__':main()
