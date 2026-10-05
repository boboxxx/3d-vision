"""Actual LDPC/QAM; disclosed NR variant, no detector or learned model."""
import copy
import hashlib
import math

import numpy as np
from scipy.special import logsumexp
import torch
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import qam

CONFIGURATIONS = {'ldpc_2_3_qam64': (1296,1944,6), 'ldpc_1_2_qam256': (972,1944,8)}


def require(value,message):
    if not value: raise ValueError(message)


def array_record(a):
    a=np.ascontiguousarray(a)
    return dict(shape=list(a.shape),dtype=a.dtype.str,sha256=hashlib.sha256(a.tobytes()).hexdigest())


def labels(bits_per_symbol):
    return ((np.arange(2**bits_per_symbol)[:,None] >> np.arange(bits_per_symbol-1,-1,-1)) & 1).astype(np.uint8)


def app_demapper(received,noise_variance,points,bit_labels,chunk_size=4096):
    received=np.asarray(received,dtype=np.complex128)
    variance=np.broadcast_to(np.asarray(noise_variance,dtype=np.float64),received.shape)
    require(received.ndim==1 and len(received)>0,'nonempty complex symbol vector')
    require(np.isfinite(received).all() and np.isfinite(variance).all() and (variance>0).all(),'finite positive demapper variance')
    require(points.ndim==1 and bit_labels.shape[0]==len(points),'constellation/label alignment')
    llr=np.empty((len(received),bit_labels.shape[1]),dtype=np.float64)
    for start in range(0,len(received),chunk_size):
        stop=min(start+chunk_size,len(received))
        scores=-np.abs(received[start:stop,None]-points[None,:])**2/variance[start:stop,None]
        for bit in range(bit_labels.shape[1]):
            llr[start:stop,bit]=logsumexp(scores[:,bit_labels[:,bit]==1],axis=1)-logsumexp(scores[:,bit_labels[:,bit]==0],axis=1)
    require(np.isfinite(llr).all(),'nonfinite APP LLR')
    return llr


def physical_channel(symbols,kind,snr_db,noise_rng,fading_rng):
    require(kind in ('identity','awgn','rayleigh'),'unknown physical channel')
    require(math.isfinite(snr_db),'finite SNR')
    require(noise_rng is not fading_rng and all(isinstance(r.bit_generator,np.random.PCG64) for r in (noise_rng,fading_rng)), 'separate PCG64 required')
    x=np.asarray(symbols,dtype=np.complex128)
    require(x.ndim==1 and len(x)>0 and np.isfinite(x).all(),'finite nonempty symbols')
    before=dict(noise=copy.deepcopy(noise_rng.bit_generator.state),fading=copy.deepcopy(fading_rng.bit_generator.state))
    n0=1e-6 if kind=='identity' else 10.0**(-snr_db/10.0)
    require(math.isfinite(n0) and n0>0,'finite positive nominal N0')
    noise=np.zeros_like(x); h=np.ones_like(x)
    if kind!='identity':
        draw=noise_rng.standard_normal((len(x),2))*math.sqrt(n0/2)
        noise=draw[:,0]+1j*draw[:,1]
        if kind=='rayleigh':
            draw=fading_rng.standard_normal((len(x),2))/math.sqrt(2)
            h=draw[:,0]+1j*draw[:,1]
    require((np.abs(h)>0).all(),'exact zero cannot be ZF equalized')
    received=(h*x+noise)/h
    variance=n0/np.abs(h)**2
    require(np.isfinite(received).all() and np.isfinite(variance).all(),'nonfinite exact ZF result')
    record=dict(channel=kind,SNR_dB=float(snr_db),nominal_population_Es=1.0,N0=n0,
                identity_demapper_variance_only=kind=='identity',actual_energy=float(np.sum(np.abs(x)**2)),
                actual_mean_Es=float(np.mean(np.abs(x)**2)),uses=len(x),pilot_uses=0,
                equalizer='exact_ZF',CSI='perfect' if kind=='rayleigh' else 'not_needed',
                coherence_symbols=1 if kind=='rayleigh' else None,
                transmitted=array_record(x),noise=array_record(noise),fading=array_record(h),
                received=array_record(received),effective_noise_variance=array_record(variance),
                rng_before=before,rng_after=dict(noise=copy.deepcopy(noise_rng.bit_generator.state),
                fading=copy.deepcopy(fading_rng.bit_generator.state)))
    return received,variance,record,dict(transmitted=x,received=received,noise=noise,fading=h,effective_noise_variance=variance)


class DigitalTransceiver:
    def __init__(self,configuration):
        require(configuration in CONFIGURATIONS,'unknown locked configuration')
        self.configuration=configuration
        self.k,self.n,self.bps=CONFIGURATIONS[configuration]
        self.encoder=LDPC5GEncoder(self.k,self.n,num_bits_per_symbol=self.bps,precision='double',device='cpu')
        self.decoder=LDPC5GDecoder(self.encoder,cn_update='boxplus-phi',vn_update='sum',cn_schedule='flooding',
                                  hard_out=True,return_infobits=True,num_iter=20,llr_max=20.0,
                                  prune_pcm=True,precision='double',device='cpu')
        self.points=qam(self.bps,normalize=True,precision='double')
        self.bit_labels=labels(self.bps)

    def identity(self):
        pcm=self.encoder.pcm
        return dict(configuration=self.configuration,k=self.k,n=self.n,bits_per_symbol=self.bps,
                    code_family='NR_38.212_declared_ambiguity_variant',basegraph=self.encoder._bg,
                    lifting_factor=self.encoder.z,k_ldpc=self.encoder.k_ldpc,n_ldpc=self.encoder.n_ldpc,
                    k_filler=self.encoder.k_filler,punctured_first_bits=2*self.encoder.z,
                    RV=0,output_interleaver=array_record(self.encoder.out_int.cpu().numpy()),
                    pcm_shape=list(pcm.shape),pcm_indptr=array_record(pcm.indptr),
                    pcm_indices=array_record(pcm.indices),pcm_values=array_record(pcm.data),
                    points=array_record(self.points),labels=array_record(self.bit_labels),
                    decoder_iterations=20,decoder_llr_clip=20.0,decoder_rule='boxplus-phi',
                    decoder_schedule='flooding',precision='double',device='cpu')

    def transmit(self,payload,*,kind,snr_db=10.0,noise_rng,fading_rng):
        require(isinstance(payload,bytes) and len(payload)>0,'nonempty actual bytes required')
        bits=np.unpackbits(np.frombuffer(payload,dtype=np.uint8),bitorder='big')
        blocks=(len(bits)+self.k-1)//self.k
        padded=np.pad(bits,(0,blocks*self.k-len(bits))).reshape(blocks,self.k)
        source=torch.from_numpy(padded.astype(np.float64))
        with torch.no_grad():coded=self.encoder(source).cpu().numpy().astype(np.uint8)
        require(coded.shape==(blocks,self.n) and ((coded==0)|(coded==1)).all(),'actual coded bit shape')
        grouped=coded.reshape(-1,self.bps)
        indices=(grouped*(1 << np.arange(self.bps-1,-1,-1))).sum(1)
        symbols=self.points[indices]
        received,variance,channel,arrays=physical_channel(symbols,kind,snr_db,noise_rng,fading_rng)
        llr=app_demapper(received,variance,self.points,self.bit_labels).reshape(blocks,self.n)
        decoded=[]
        with torch.no_grad():
            for start in range(0,blocks,32):
                decoded.append(self.decoder(torch.from_numpy(llr[start:start+32])).cpu().numpy().astype(np.uint8))
        decoded=np.concatenate(decoded)
        require(decoded.shape==padded.shape and ((decoded==0)|(decoded==1)).all(),'actual decoded source shape')
        source_bits=decoded.reshape(-1)[:len(bits)]
        recovered=np.packbits(source_bits,bitorder='big').tobytes()
        record=dict(scope='actual_digital_packet_no_detector_AP',identity=self.identity(),
                    source_bytes=len(payload),source_bits=len(bits),blocks=blocks,
                    source_padding_bits=blocks*self.k-len(bits),coded_bits=int(coded.size),QAM_padding_bits=0,
                    exact_payload_recovered=recovered==payload,source_bit_errors=int(np.count_nonzero(source_bits!=bits)),
                    source_block_errors=int(np.count_nonzero(np.any(decoded!=padded,axis=1))),
                    source_padding_valid=bool(not np.any(decoded.reshape(-1)[len(bits):])),
                    source_sha256=hashlib.sha256(payload).hexdigest(),received_sha256=hashlib.sha256(recovered).hexdigest(),
                    coded=array_record(coded),decoded=array_record(decoded),channel=channel,
                    reliable_uses=0,reliable_energy=0.0,source_length_signaling='engineering_public_metadata_main_framing_unresolved')
        arrays.update(coded=coded,decoded=decoded,source_padded=padded,llr=llr)
        return recovered,record,arrays
