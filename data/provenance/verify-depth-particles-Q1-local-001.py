"""Complete transferred Q1 physical records; independent CDF-domain W1 replay."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
NAMES=['narrow_near','narrow_middle','narrow_far','uniform','broad_middle','broad_far',
       'bimodal_50_50','bimodal_35_65','far_minority_5pct','far_minority_10pct',
       'far_minority_20pct','far_minority_40pct','trimodal_equal']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def quantile(p,e,u):
    p=p/p.sum();total=0.
    for i,m in enumerate(p):
        if m==0:continue
        if u<=total+m+1e-16:return float(e[i]+min(max((u-total)/m,0),1)*(e[i+1]-e[i]))
        total+=m
    return float(e[np.flatnonzero(p>0)[-1]+1])

def histogram_cdf(p,e,z):
    p=p/p.sum();i=np.searchsorted(e,z,side='right')-1;i=i.clip(0,len(p)-1)
    return np.r_[0,np.cumsum(p)[:-1]][i]+p[i]*(z-e[i])/(e[i+1]-e[i])

def linear_abs_integral(x,y,width):
    result=np.empty_like(x,dtype=np.float64);same=x*y>=0
    result[same]=width[same]*(np.abs(x[same])+np.abs(y[same]))/2
    opposite=~same
    result[opposite]=width[opposite]*(x[opposite]**2+y[opposite]**2)/(2*(np.abs(x[opposite])+np.abs(y[opposite])))
    return float(result.sum())

def hist_W1(p,e,a,weights):
    # Spatial CDF integration, independent of production quantile-domain splitting.
    breaks=np.unique(np.r_[e,a]);left=breaks[:-1];right=breaks[1:];mid=(left+right)/2
    atomic=((a[None,:]<=mid[:,None])*weights[None,:]).sum(-1)
    return linear_abs_integral(histogram_cdf(p,e,left)-atomic,histogram_cdf(p,e,right)-atomic,right-left)

def laws_W1(a,w,b,v):
    breaks=np.unique(np.r_[a,b]);mid=(breaks[:-1]+breaks[1:])/2
    x=((a[None,:]<=mid[:,None])*w[None,:]).sum(-1)
    y=((b[None,:]<=mid[:,None])*v[None,:]).sum(-1)
    return float(np.sum(np.abs(x-y)*np.diff(breaks)))

def hist_distance(p,q,e):
    return linear_abs_integral(histogram_cdf(p,e,e[:-1])-histogram_cdf(q,e,e[:-1]),
                               histogram_cdf(p,e,e[1:])-histogram_cdf(q,e,e[1:]),np.diff(e))

def scalar_projection(q,ordered):
    blocks=[[float(v),1] for v in q[:ordered]]
    i=0
    while i<len(blocks)-1:
        if blocks[i][0]/blocks[i][1]>blocks[i+1][0]/blocks[i+1][1]:
            blocks[i:i+2]=[[blocks[i][0]+blocks[i+1][0],blocks[i][1]+blocks[i+1][1]]];i=max(i-1,0)
        else:i+=1
    values=[s/n for s,n in blocks for _ in range(n)]
    if ordered==2:values.append(float(q[2]))
    return np.clip(values,-1,1)

def global_sample_objective(p,e):
    n=240000;u=(np.arange(n)+.5)/n;p=p/p.sum();cdf=p.cumsum();cdf[-1]=1
    i=np.searchsorted(cdf,u);prev=np.r_[0,cdf[:-1]][i]
    x=e[i]+(u-prev)/p[i]*np.diff(e)[i];prefix=np.r_[0,x.cumsum()];split=np.arange(1,n)
    k=(split-1)//2;v=x[k]
    first=v*k-prefix[k]+prefix[split]-prefix[k+1]-v*(split-k-1)
    j=(split+n-1)//2;v=x[j]
    second=v*(j-split)-(prefix[j]-prefix[split])+prefix[n]-prefix[j+1]-v*(n-j-1)
    return float((first+second).min()/n)

def assert_close(a,b,atol=2e-10):
    np.testing.assert_allclose(a,b,atol=atol,rtol=0)
    return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))

def main():
    out=ROOT/'data/provenance/depth-particles-Q1-complete-local-verification-001.json';assert not out.exists()
    artifacts={};reports={};cells=0;candidate_records=0;stats=0;max_hist=max_law=max_coordinate=0.;global_error=0.
    cross_arrays={};prior_replay=0
    for host in ('local','sheng'):
        folder=ROOT/'data/engineering'/('depth-particles-Q1-'+host+'-CPU-001')
        rp=folder/'report.json';raw=folder/'complete_arrays.npz';report=json.loads(rp.read_text())
        assert report['state']=='passed_Q1_matched_synthetic_mode_mass_transport_and_source_stability_CPU'
        assert report['host']==host and report['source_distributions']==13 and report['methods']==2
        assert report['conditions_count']==7 and report['repeats_each']==64 and report['arrays_sha256']==sha(raw)
        assert report['no_fitted_learned_parameters'] and report['CPU_threads']==2
        for name,digest in report['sources'].items():assert sha(ROOT/name)==digest
        assert len(report['sources'])==6
        artifacts.update({str(p.relative_to(ROOT)):sha(p) for p in (rp,raw)})
        reports[host]=report
        with np.load(raw,allow_pickle=False) as z,np.load(ROOT/'data/engineering'/('depth-particles-Q0-'+host+'-CPU-001')/'complete_arrays.npz',allow_pickle=False) as old:
            assert sorted(z.files)==report['arrays_keys'];p=z['probabilities'];e=z['depth_edges'];span=e[-1]-e[0]
            assert_close(e,2+.8*np.arange(73),1e-12);assert p.shape==(13,72)
            assert_close(p[:12],old['probabilities'],1e-15)
            last=np.zeros(72);last[[7,37,67]]=1/3;assert_close(p[-1],last,1e-15)
            rng=np.random.Generator(np.random.PCG64(17));noise=rng.normal(size=(64,4));fade=(rng.normal(size=(64,2))+1j*rng.normal(size=(64,2)))/np.sqrt(2)
            assert_close(z['common_standard_noise'],noise,0);assert_close(z['common_Rayleigh_fade'],fade,0)
            assert_close(z['common_standard_noise'],old['common_standard_noise'],0)
            assert_close(z['common_Rayleigh_fade'],old['common_Rayleigh_fade'],0)
            a0=z['Q0_atoms'];a1=z['Q1_atoms'];m1=z['Q1_first_mass']
            independent0=np.array([[quantile(source,e,u) for u in (1/6,.5,5/6)] for source in p]);assert_close(a0,independent0,2e-12)
            for index,source in enumerate(p):
                m=float(m1[index]);a=a1[index]
                assert_close(a,[quantile(source,e,m/2),quantile(source,e,(1+m)/2)],2e-12)
                candidates=z['fit_'+str(index)+'_candidate_splits'];costs=z['fit_'+str(index)+'_candidate_objectives']
                assert np.all(np.diff(candidates)>0) and candidates[0]==0 and candidates[-1]==1
                actual=[]
                for split in candidates:
                    points=np.array([quantile(source,e,split/2),quantile(source,e,(1+split)/2)])
                    actual.append(hist_W1(source,e,points,np.array([split,1-split])))
                max_hist=max(max_hist,assert_close(actual,costs));candidate_records+=len(candidates)
                value=hist_W1(source,e,a,np.array([m,1-m]));assert value<=min(actual)+1e-11+2e-10
                eligible=candidates[costs<=costs.min()+1e-11];assert m==eligible[0]
                difference=abs(value-global_sample_objective(source,e));assert difference<=span/(2*240000)+2e-10;global_error=max(global_error,difference)
            source_coordinates={'Q0':z['Q0_normalized_coordinates'],'Q1':z['Q1_normalized_coordinates']}
            assert_close(source_coordinates['Q0'],2*(a0-e[0])/span-1,2e-12)
            assert_close(source_coordinates['Q1'],np.c_[2*(a1-e[0])/span-1,2*m1-1],2e-12)
            expected_conditions=[('identity',None)]+[(c,s) for c in ('awgn','rayleigh') for s in (6,10,18)]
            for condition,(channel,snr) in zip(report['conditions'],expected_conditions):
                assert (condition['channel'],condition['SNR_dB'])==(channel,snr)
                assert condition['perfect_CSI']==(channel=='rayleigh') and not condition['fade_floor']
                tag=channel if snr is None else channel+str(snr)
                n=np.zeros_like(noise) if snr is None else noise*np.sqrt(10**(-snr/10)/2)
                h=fade if channel=='rayleigh' else np.ones_like(fade)
                assert_close(z[tag+'_noise_real'],n,0);assert_close(z[tag+'_fade'],h,0)
                nc=np.c_[n[:,0]+1j*n[:,1],n[:,2]+1j*n[:,3]]
                for method in ('Q0','Q1'):
                    coord=source_coordinates[method];real=np.c_[coord,np.ones(13)]*np.sqrt(2/(1+(coord**2).sum(-1,keepdims=True)))
                    tx=np.c_[real[:,0]+1j*real[:,1],real[:,2]+1j*real[:,3]];key=tag+'_'+method
                    assert_close(z[key+'_transmitted_complex'],tx,2e-12)
                    raw_rx=tx[:,None,:]*h[None,:,:]+nc[None,:,:];rx=raw_rx/h[None,:,:]
                    real_rx=np.stack((rx[...,0].real,rx[...,0].imag,rx[...,1].real,rx[...,1].imag),-1)
                    assert_close(z[key+'_raw_received'],raw_rx,2e-12);assert_close(z[key+'_equalized_real'],real_rx,2e-12)
                    energy=float((abs(tx)**2).sum()*64);budget=condition['methods'][method]
                    assert budget['complex_uses']==1664 and budget['mean_Es']==budget['total_energy']/1664
                    assert_close(energy,budget['total_energy'],2e-10);assert_close(energy,1664,2e-10)
                    decoded=z[key+'_decoded_coordinates'];atoms=z[key+'_decoded_atoms'];mass=z[key+'_decoded_first_mass']
                    all_hist=np.empty((13,64));all_law=np.empty((13,64))
                    for i,source in enumerate(p):
                        src_a=a0[i] if method=='Q0' else a1[i]
                        src_w=np.ones(3)/3 if method=='Q0' else np.array([m1[i],1-m1[i]])
                        intrinsic=hist_W1(source,e,src_a,src_w)
                        assert_close(intrinsic,report['intrinsic_W1_m'][method][NAMES[i]])
                        for j in range(64):
                            expected=scalar_projection(real_rx[i,j,:3]/max(real_rx[i,j,3],2**-.5),3 if method=='Q0' else 2)
                            max_coordinate=max(max_coordinate,assert_close(decoded[i,j],expected,2e-12))
                            assert_close(atoms[i,j],e[0]+(decoded[i,j,:(3 if method=='Q0' else 2)]+1)*span/2,2e-12)
                            weights=np.ones(3)/3 if method=='Q0' else np.array([mass[i,j],1-mass[i,j]])
                            assert_close(mass[i,j],1/3 if method=='Q0' else (decoded[i,j,2]+1)/2,2e-12)
                            all_hist[i,j]=hist_W1(source,e,atoms[i,j],weights)
                            all_law[i,j]=laws_W1(src_a,src_w,atoms[i,j],weights);cells+=1
                        row=budget['rows'][i];assert row['name']==NAMES[i]
                        assert_close([row['intrinsic_W1_m'],row['channel_W1_mean_m'],row['full_W1_mean_m'],row['full_W1_max_m']],
                            [intrinsic,all_law[i].mean(),all_hist[i].mean(),all_hist[i].max()]);stats+=1
                    max_hist=max(max_hist,assert_close(all_hist,z[key+'_full_W1']))
                    max_law=max(max_law,assert_close(all_law,z[key+'_channel_W1']))
                    if method=='Q0':
                        for suffix,old_suffix in [('decoded_atoms','decoded_particles'),('channel_W1','channel_W1'),('full_W1','full_histogram_W1')]:
                            assert_close(z[key+'_'+suffix][:12],old[tag+'_'+old_suffix],2e-12)
                        prior_replay+=12*64
            for row in report['stability']:
                key='stability_'+row['family']+'_'+str(row['epsilon']);pair=z[key+'_source_pair']
                assert_close(row['source_W1_m'],hist_distance(pair[0],pair[1],e))
                for i,source in enumerate(pair):
                    q=z[key+'_Q1_coordinates'][i];m=(q[2]+1)/2;points=e[0]+(q[:2]+1)*span/2
                    assert_close(m,row['Q1_splits'][i]);value=hist_W1(source,e,points,np.array([m,1-m]))
                    assert_close(value,row['Q1_intrinsic_W1_m'][i])
                    assert abs(value-global_sample_objective(source,e))<=span/(2*240000)+2e-10
                    for method,q in [('Q1',q),('Q0',2*(np.array([quantile(source,e,u) for u in (1/6,.5,5/6)])-e[0])/span-1)]:
                        tx=np.r_[q,1]*np.sqrt(2/(1+(q**2).sum()));assert_close(tx,z[key+'_'+method+'_tx'][i],2e-12)
                for method in ('Q0','Q1'):assert_close(row[method+'_carrier_distance'],np.linalg.norm(z[key+'_'+method+'_tx'][0]-z[key+'_'+method+'_tx'][1]),2e-12)
            cross_arrays[host]={key:z[key].copy() for key in z.files if not key.startswith('fit_')}
    assert reports['local']['sources']==reports['sheng']['sources']
    cross=max(assert_close(cross_arrays['local'][k],cross_arrays['sheng'][k],2e-11) for k in cross_arrays['local'])
    assert cells==23296 and stats==364 and prior_replay==10752
    result=dict(state='passed_all23296_transferred_Q1_Q0_cells_independent_CDF_replay',checked_unix=time.time(),
        artifacts_sha256=artifacts,artifacts_verified=4,source_conditions_statistics_verified=stats,cells_verified=cells,
        all_actual_fit_candidate_records_checked=candidate_records,global240000_sample_max_disagreement=global_error,
        previous_Q0_cells_replayed=prior_replay,sources=reports['local']['sources'],
        independent_histogram_W1_max_error=max_hist,independent_law_W1_max_error=max_law,
        independent_scalar_projection_max_error=max_coordinate,cross_host_physical_max_error=cross,
        verifier_sha256=sha(__file__),limitation='Complete synthetic files/physical/source-stability records; no native teacher, differentiable fitting/task adapter, KITTI, GPU, AP or novelty claim')
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ['state','cells_verified','all_actual_fit_candidate_records_checked','independent_histogram_W1_max_error','cross_host_physical_max_error']}))

if __name__=='__main__':main()
