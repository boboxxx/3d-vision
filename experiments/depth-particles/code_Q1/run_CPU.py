"""Matched Q0/Q1 transport and predeclared modal/source-stability probes."""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import Bounds,LinearConstraint,minimize
import torch
import representation as r

ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'experiments/depth-particles/code'

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

q0=module('equal_mass_Q0',OLD/'representation.py')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def sources():
    files=[ROOT/'experiments/depth-particles/protocol-002.md',ROOT/'experiments/depth-particles/protocol-001.md',
           OLD/'representation.py',OLD/'run_CPU.py',*Path(__file__).resolve().parent.glob('*.py')]
    return {str(p.relative_to(ROOT)):sha(p) for p in files}

def corpus():
    # Literal unchanged Q0 distributions; extra three-mode case is predeclared.
    edges=2+.8*np.arange(73);centers=(edges[:-1]+edges[1:])/2
    names=['narrow_near','narrow_middle','narrow_far','uniform','broad_middle','broad_far',
           'bimodal_50_50','bimodal_35_65','far_minority_5pct','far_minority_10pct',
           'far_minority_20pct','far_minority_40pct','trimodal_equal']
    values=[]
    for i in (7,35,67):
        p=np.zeros(72);p[i]=1;values.append(p)
    values.append(np.ones(72)/72)
    for center,sigma in ((30,5),(50,8)):
        p=np.exp(-.5*((centers-center)/sigma)**2);values.append(p/p.sum())
    for m in (.5,.65,.05,.1,.2,.4):
        p=np.zeros(72);p[7]=1-m;p[67]=m;values.append(p)
    p=np.zeros(72);p[[7,37,67]]=1/3;values.append(p)
    return names,np.array(values),edges

def discrete_global_optimum(p,edges,N=240000):
    # Independent finite-source exhaustive median partitions using prefix sums.
    levels=(np.arange(N)+.5)/N;cum=p.cumsum();cum[-1]=1
    ids=np.searchsorted(cum,levels);before=np.r_[0,cum[:-1]][ids]
    points=edges[ids]+(levels-before)/p[ids]*np.diff(edges)[ids]
    prefix=np.r_[0,np.cumsum(points)];k=np.arange(1,N)
    def cost(start,end):
        med=(start+end-1)//2;v=points[med]
        return v*(med-start)-(prefix[med]-prefix[start])+(prefix[end]-prefix[med+1])-v*(end-med-1)
    costs=(cost(np.zeros_like(k),k)+cost(k,np.full_like(k,N)))/N
    index=int(costs.argmin());return float(costs[index]),float(k[index]/N)

def checks(p,edges,fits):
    rng=np.random.Generator(np.random.PCG64(17));span=edges[-1]-edges[0]
    discrepancies=[];references=[]
    for source,fit in zip(p,fits):
        objective,split=discrete_global_optimum(source,edges)
        difference=abs(fit['W1']-objective);assert difference<=span/(2*240000)+2e-10
        discrepancies.append(difference);references.append(dict(discrete_W1=objective,discrete_split=split))
    probes=rng.normal(0,3,size=(32,3));projected=r.project(torch.from_numpy(probes)).numpy();error=0.
    constraint=LinearConstraint(np.array([[-1.,1.,0.]]),0,np.inf)
    for x,y in zip(probes,projected):
        result=minimize(lambda t:.5*np.sum((t-x)**2),np.zeros(3),jac=lambda t:t-x,
            bounds=Bounds(-np.ones(3),np.ones(3)),constraints=[constraint],method='SLSQP',
            options={'ftol':1e-13,'maxiter':200})
        assert result.success;error=max(error,float(np.max(np.abs(result.x-y))))
    assert error<=2e-7
    pairs=list(itertools.combinations_with_replacement(np.linspace(-1,1,11),2))
    grid=torch.tensor([[a,b,m] for a,b in pairs for m in np.linspace(-1,1,11)],dtype=torch.float64)
    noise=torch.from_numpy(rng.normal(0,2,size=(len(grid),32,4)))
    decoded=r.decode(r.encode(grid)[:,None,:]+noise)
    excess=float(((decoded-grid[:,None,:]).norm(dim=-1)-2*np.sqrt(2)*noise.norm(dim=-1)).max())
    assert excess<=2e-12 and len(grid)*32==23232
    a,m=r.unpack(grid.numpy(),edges[0],edges[-1]);b,n=r.unpack(decoded.numpy(),edges[0],edges[-1])
    law=r.law_W1(a[:,None,:],m[:,None],b,n)
    law_excess=float(np.max(law-2*span*noise.norm(dim=-1).numpy()));assert law_excess<=2e-11
    q=r.coordinates(np.array([f['atoms'] for f in fits]),np.array([f['mass'] for f in fits]),edges[0],edges[-1])
    tx=r.encode(torch.from_numpy(q));inverse_error=float((r.decode(tx)-torch.from_numpy(q)).abs().max())
    energy_error=float((tx.square().sum(-1)-2).abs().max());assert inverse_error<=1e-12 and energy_error<=1e-12
    probe=torch.tensor([[-.7,.3,.1],[-.5,.8,-.6]],dtype=torch.float64,requires_grad=True)
    passed=torch.autograd.gradcheck(lambda x:r.decode(r.encode(x)),(probe,),eps=1e-6,atol=1e-7,rtol=1e-5);assert passed
    return dict(global_discrete_N=240000,global_objective_max_disagreement=max(discrepancies),
        global_objective_allowed_disagreement=span/(2*240000)+2e-10,global_references=references,
        projection_cases=32,projection_optimizer_max_error=error,bound_cases=23232,
        coordinate_bound_max_excess=excess,law_W1_bound_max_excess=law_excess,
        noiseless_error=inverse_error,energy_error=energy_error,carrier_gradient_passed=passed,
        optimal_fitting_gradient_claimed=False),q,tx.numpy()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--host',choices=('local','sheng'),required=True);args=parser.parse_args()
    torch.set_num_threads(2);folder=ROOT/'data/engineering'/('depth-particles-Q1-'+args.host+'-CPU-001')
    assert not folder.exists();folder.mkdir();started=time.time();identity=sources()
    names,p,edges=corpus();fits=[r.fit_two(source,edges) for source in p]
    check,q1,tx1=checks(p,edges,fits);a1=np.array([f['atoms'] for f in fits]);m1=np.array([f['mass'] for f in fits])
    a0=q0.particles(torch.from_numpy(p),torch.from_numpy(edges)).numpy()
    z0=q0.scale_depth(torch.from_numpy(a0),edges[0],edges[-1]);tx0=q0.encode(z0).numpy()
    intrinsic={'Q0':q0.histogram_W1(p,edges,a0),'Q1':np.array([f['W1'] for f in fits])}
    arrays=dict(probabilities=p,depth_edges=edges,Q0_atoms=a0,Q1_atoms=a1,Q1_first_mass=m1,
        Q0_normalized_coordinates=z0.numpy(),Q1_normalized_coordinates=q1)
    for i,f in enumerate(fits):
        arrays['fit_'+str(i)+'_candidate_splits']=f['candidate_splits']
        arrays['fit_'+str(i)+'_candidate_objectives']=f['candidate_objectives']
    rng=np.random.Generator(np.random.PCG64(17));noise0=rng.normal(size=(64,4))
    fades0=(rng.normal(size=(64,2))+1j*rng.normal(size=(64,2)))/np.sqrt(2)
    arrays.update(common_standard_noise=noise0,common_Rayleigh_fade=fades0)
    conditions=[]
    for channel,snr in [('identity',None)]+[(c,s) for c in ('awgn','rayleigh') for s in (6,10,18)]:
        tag=channel if snr is None else channel+str(snr)
        noise=np.zeros_like(noise0) if snr is None else noise0*np.sqrt(10**(-snr/10)/2)
        fade=fades0 if channel=='rayleigh' else np.ones_like(fades0)
        nc=np.stack((noise[:,0]+1j*noise[:,1],noise[:,2]+1j*noise[:,3]),-1)
        arrays.update({tag+'_noise_real':noise,tag+'_fade':fade});methods={}
        for method,tx in [('Q0',tx0),('Q1',tx1)]:
            symbols=np.stack((tx[:,0]+1j*tx[:,1],tx[:,2]+1j*tx[:,3]),-1)
            raw=symbols[:,None,:]*fade[None,:,:]+nc[None,:,:];equalized=raw/fade[None,:,:]
            received=np.stack((equalized[...,0].real,equalized[...,0].imag,equalized[...,1].real,equalized[...,1].imag),-1)
            if method=='Q0':
                decoded=q0.decode(torch.from_numpy(received)).numpy();atoms=q0.unscale_depth(decoded,edges[0],edges[-1])
                chan=np.abs(atoms-a0[:,None,:]).mean(-1);full=q0.histogram_W1(p[:,None,:],edges,atoms)
                mass=np.full(decoded.shape[:-1],1/3)
            else:
                decoded=r.decode(torch.from_numpy(received)).numpy();atoms,mass=r.unpack(decoded,edges[0],edges[-1])
                chan=r.law_W1(a1[:,None,:],m1[:,None],atoms,mass);full=r.histogram_W1(p[:,None,:],edges,atoms,mass)
                effective=received-tx[:,None,:]
                assert np.max(np.linalg.norm(decoded-q1[:,None,:],axis=-1)-2*np.sqrt(2)*np.linalg.norm(effective,axis=-1))<=2e-12
                assert np.max(chan-2*(edges[-1]-edges[0])*np.linalg.norm(effective,axis=-1))<=2e-11
            assert np.max(full-intrinsic[method][:,None]-chan)<=2e-11 and np.isfinite(full).all()
            energy=float(np.sum(np.abs(symbols)**2)*64);uses=len(names)*64*2;assert abs(energy-uses)<=1e-10
            key=tag+'_'+method
            arrays.update({key+'_transmitted_complex':symbols,key+'_raw_received':raw,key+'_equalized_real':received,
                key+'_decoded_coordinates':decoded,key+'_decoded_atoms':atoms,key+'_decoded_first_mass':mass,
                key+'_channel_W1':chan,key+'_full_W1':full})
            methods[method]=dict(complex_uses=uses,total_energy=energy,mean_Es=energy/uses,
                rows=[dict(name=name,intrinsic_W1_m=float(base),channel_W1_mean_m=float(c.mean()),full_W1_mean_m=float(f.mean()),full_W1_max_m=float(f.max()))
                      for name,base,c,f in zip(names,intrinsic[method],chan,full)])
        conditions.append(dict(channel=channel,SNR_dB=snr,perfect_CSI=channel=='rayleigh',fade_floor=False,methods=methods))
    stability=[]
    for label,bins,base in [('two_mode',[7,67],np.array([.5,.5])),('three_mode',[7,37,67],np.ones(3)/3)]:
        for epsilon in (1e-2,1e-4,1e-6,1e-8):
            values=[]
            for sign in (-1,1):
                source=np.zeros(72);weights=base.copy();weights[0]+=sign*epsilon;weights[-1]-=sign*epsilon;source[bins]=weights
                f=r.fit_two(source,edges);coordinates=r.coordinates(f['atoms'],f['mass'],edges[0],edges[-1]);tx=r.encode(torch.from_numpy(coordinates)).numpy()
                old=q0.particles(torch.from_numpy(source),torch.from_numpy(edges));old_tx=q0.encode(q0.scale_depth(old,edges[0],edges[-1])).numpy()
                values.append(dict(source=source,fit=f,coordinates=coordinates,tx=tx,old_tx=old_tx))
            key='stability_'+label+'_'+str(epsilon)
            arrays.update({key+'_source_pair':np.array([v['source'] for v in values]),key+'_Q1_coordinates':np.array([v['coordinates'] for v in values]),
                key+'_Q1_tx':np.array([v['tx'] for v in values]),key+'_Q0_tx':np.array([v['old_tx'] for v in values])})
            stability.append(dict(family=label,epsilon=epsilon,source_W1_m=r.source_W1(values[0]['source'],values[1]['source'],edges),
                Q0_carrier_distance=float(np.linalg.norm(values[0]['old_tx']-values[1]['old_tx'])),
                Q1_carrier_distance=float(np.linalg.norm(values[0]['tx']-values[1]['tx'])),
                Q1_splits=[v['fit']['mass'] for v in values],Q1_intrinsic_W1_m=[v['fit']['W1'] for v in values]))
    assert sources()==identity;raw=folder/'complete_arrays.npz';np.savez(raw,**arrays)
    result=dict(state='passed_Q1_matched_synthetic_mode_mass_transport_and_source_stability_CPU',host=args.host,
        sources=identity,started_unix=started,ended_unix=time.time(),checks=check,conditions=conditions,stability=stability,
        source_distributions=len(names),methods=2,conditions_count=7,repeats_each=64,
        intrinsic_W1_m={method:{name:float(value) for name,value in zip(names,dist)} for method,dist in intrinsic.items()},
        arrays_sha256=sha(raw),arrays_keys=sorted(arrays),no_fitted_learned_parameters=True,
        torch_version=torch.__version__,numpy_version=np.__version__,CPU_threads=2,
        limitation='Synthetic piecewise-uniform optimal quantization and charged carrier only; hard source fit may be discontinuous, no native teacher/KITTI/GPU/differentiable task adapter/AP/novelty claim')
    with (folder/'report.json').open('x') as out:json.dump(result,out,indent=2,allow_nan=False)
    print(json.dumps(dict(state=result['state'],intrinsic_W1_m=result['intrinsic_W1_m'],stability=stability)))

if __name__=='__main__':main()
