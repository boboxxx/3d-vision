"""Independent spatial-CDF replay and direct partition costs, every native cell."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
FOLDER=ROOT/'data/engineering/depth-particles-native-Q2-sheng-CPU-001'
FIXTURES=Path('/mnt/d/paper6/runs/geometry-risk-native-engineering-001')
LOCK={'000000':('cbb10d56055ec6c540b5f103d25a904e67d80c06300580fe6ee3104fa927fd17',118431696),
      '000003':('402f4c91fc9a26100d8e20f14a16b6b9215a65d93d39858d6986ad8299062a58',117746968)}
KEYS=['left_probability','right_probability','left_valid','right_valid','disparity_bins','P2','P3','valid_image_shape']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def cdf_distance(p,z,w,a):
    z=np.broadcast_to(z,p.shape)
    points=np.concatenate((z,a),-1); delta=np.concatenate((p,-w),-1)
    order=np.argsort(points,axis=-1,kind='stable')
    points=np.take_along_axis(points,order,-1)
    cumulative=np.take_along_axis(delta,order,-1).cumsum(-1)
    return (np.abs(cumulative[:,:-1])*np.diff(points,axis=-1)).sum(-1)

def project(x,method):
    if method=='Q1':
        y=x.copy();bad=x[:,0]>x[:,1];y[bad,:2]=x[bad,:2].mean(-1)[:,None]
        return y.clip(-1,1)
    y=x.copy();a,b,c=x.T
    first=a>b;last=(~first)&(b>c)
    y[first,:2]=((a+b)/2)[first,None]
    y[last,1:]=((b+c)/2)[last,None]
    allpool=(first&(((a+b)/2)>c))|(last&(a>((b+c)/2)))
    y[allpool]=x[allpool].mean(-1)[:,None]
    return y.clip(-1,1)

def summary(x):
    return dict(mean_W1_m=float(x.mean()),median_W1_m=float(np.median(x)),
        max_W1_m=float(x.max()),p95_W1_m=float(np.quantile(x,.95)),count=len(x))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--native',action='store_true');args=ap.parse_args()
    report=json.loads((FOLDER/'report.json').read_text())
    assert report['state']=='passed_complete_native_cached_posterior_transport_pending_independent_local_verification'
    assert report['read_keys']==KEYS and report['seed']==2806 and report['bounds_m']==[0.,128.]
    for p,h in report['sources'].items():assert sha(ROOT/p)==h,(p,'source')
    rng=np.random.Generator(np.random.PCG64(2806));artifacts={str((FOLDER/'report.json').relative_to(ROOT)):sha(FOLDER/'report.json')}
    errors={};pooled={};native={};grid=validtotal=uses=fitcandidates=0
    def close(got,expected,label,tol=4e-11):
        assert got.shape==expected.shape,label
        assert np.array_equal(np.isnan(got),np.isnan(expected)),label
        finite=np.isfinite(expected);assert np.array_equal(np.isfinite(got),finite),label
        e=float(np.max(np.abs(got[finite]-expected[finite]),initial=0));errors[label]=max(errors.get(label,0),e)
        assert e<=tol,(label,e)
    for entry in report['views']:
        path=ROOT/entry['artifact'];assert sha(path)==entry['sha256'] and path.stat().st_size==entry['bytes']
        artifacts[entry['artifact']]=sha(path)
        with np.load(path,allow_pickle=False) as archive:data={k:archive[k] for k in archive.files}
        frame,view=entry['tag'].split('-');praw=data['probability'];v=data['valid'];z=data['depth_support'];n=len(v)
        assert praw.shape==(24960,48) and v.shape==(24960,) and v.dtype==np.bool_
        assert n==entry['grid_cells'] and int(v.sum())==entry['valid_cells']
        assert (praw[~v]==0).all() and np.isfinite(praw).all() and (praw>=0).all()
        assert np.max(np.abs(praw[v].sum(-1)-1))<2e-14
        p=praw[v]/praw[v].sum(-1,keepdims=True);c=p.cumsum(-1);c[:,-1]=1.
        assert data['P2'].dtype==data['P3'].dtype==np.float32
        fb=float(abs(data['P2'][0,3]-data['P3'][0,3]))
        assert fb==report['fixtures'][frame]['focal_baseline']
        close(z,(fb/(4*np.arange(1,49)))[::-1],'calibrated_support',1e-13)
        assert np.array_equal(data['disparity_bins'],4*np.arange(1,49)) and 0<z[0]<z[-1]<128
        if args.native:
            source=FIXTURES/(frame+'-clean-fixture.npz');h,size=LOCK[frame]
            assert sha(source)==h and source.stat().st_size==size
            with np.load(source,allow_pickle=False) as a:clean={k:a[k].copy() for k in KEYS}
            assert np.array_equal(praw,clean[view+'_probability'][::-1].reshape(48,-1).T)
            assert np.array_equal(v,clean[view+'_valid'].reshape(-1))
            for k in ('P2','P3','disparity_bins','valid_image_shape'):assert np.array_equal(data[k],clean[k])
            assert sha(source)==h
            native[entry['tag']]=dict(archive_sha256=h,bytes=size,full_probability_and_metadata_equal=True,read_keys=KEYS)
        elif frame not in native:
            recorded=json.loads((ROOT/'data/provenance/depth-particles-native-Q2-native-verification-001.json').read_text())
            assert recorded['state']=='passed_all_native_Q2_cells_independent_spatial_CDF_and_direct_partition_replay'
            assert recorded['artifacts_sha256']=={str((FOLDER/'report.json').relative_to(ROOT)):sha(FOLDER/'report.json'),**{e['artifact']:e['sha256'] for e in report['views']}}
            native[frame]=dict(sealed_native_verification_sha256=sha(ROOT/'data/provenance/depth-particles-native-Q2-native-verification-001.json'))
        # Independent absolute-distance median costs for every split and every source.
        reference=np.empty_like(p);first=np.empty_like(p);second=np.empty_like(p)
        for k in range(48):
            m=c[:,k];ia=(c>=(m/2)[:,None]).argmax(-1);ib=(c>=((1+m)/2)[:,None]).argmax(-1)
            ia=np.where(m==0,ib,ia);ib=np.where(m==1,ia,ib)
            first[:,k]=z[ia];second[:,k]=z[ib]
            reference[:,k]=(p[:,:k+1]*np.abs(z[:k+1]-z[ia,None])).sum(-1)+(p[:,k+1:]*np.abs(z[k+1:]-z[ib,None])).sum(-1)
        close(data['candidate_mass'][v],c,'partition_mass',2e-14)
        close(data['candidate_W1'][v],reference,'direct_partition_cost')
        assert np.isnan(data['candidate_mass'][~v]).all() and np.isnan(data['candidate_W1'][~v]).all()
        chosen=data['selected_partition'][v].astype(int);rows=np.arange(len(p));fitcandidates+=reference.size
        assert (data['selected_partition'][~v]==-1).all() and ((chosen>=0)&(chosen<48)).all()
        assert (reference[rows,chosen]<=reference.min(-1)+1.01e-11).all()
        # Earliest tied candidate under documented tolerance, allowing only arithmetic boundary rounding.
        for k in range(48):assert not ((chosen>k)&(reference[:,k]<reference.min(-1)+.99e-11)).any()
        a1=np.stack((first[rows,chosen],second[rows,chosen]),-1)
        m=c[rows,chosen];m=np.where(a1[:,0]==a1[:,1],.5,m)
        close(data['Q1_source_atoms'][v],a1,'source_two_atoms',0)
        close(data['Q1_source_mass'][v],m,'source_mass',2e-14)
        a0=np.stack([z[(c>=level).argmax(-1)] for level in (1/6,1/2,5/6)],-1)
        close(data['Q0_source_atoms'][v],a0,'source_quantiles',0)
        expected_noise=rng.normal(size=(n,4));close(data['standard_noise'],expected_noise,'PCG_noise',0)
        fulla0=np.full((n,3),64.);fulla0[v]=a0
        fulla1=np.full((n,2),64.);fulla1[v]=a1
        fullm=np.full(n,.5);fullm[v]=m
        mu=fullm*fulla1[:,0]+(1-fullm)*fulla1[:,1]
        coords={'Q0':fulla0/64-1,'Q1':np.column_stack((fulla1/64-1,2*fullm-1)),
                'Q2':np.column_stack((fulla1[:,0],mu,fulla1[:,1]))/64-1}
        w0=np.full((len(p),3),1/3);w1=np.column_stack((m,1-m))
        intrinsic={'Q0':cdf_distance(p,z,w0,a0),'Q1':cdf_distance(p,z,w1,a1)}
        for method in ('Q0','Q1','Q2'):
            q=coords[method];close(data[method+'_coordinates'],q,'coordinates',2e-14)
            tx=np.column_stack((q,np.ones(n)))*np.sqrt(2/(1+(q*q).sum(-1)))[:,None]
            close(data[method+'_tx_real'],tx,'carrier',3e-14)
            close((tx*tx).sum(-1),np.full(n,2.),'energy',2e-14)
            source_a=a0 if method=='Q0' else a1;source_w=w0 if method=='Q0' else w1
            base=intrinsic['Q0' if method=='Q0' else 'Q1']
            if method!='Q2':
                close(data[method+'_intrinsic_W1'][v],base,'intrinsic_CDF')
                assert np.isnan(data[method+'_intrinsic_W1'][~v]).all()
                for k,value in summary(base).items():assert abs(entry['source_intrinsic'][method][k]-value)<4e-11
            for condition in ('identity','awgn10'):
                prefix=condition+'_'+method;rx=tx+(0 if condition=='identity' else expected_noise*np.sqrt(.05))
                close(data[prefix+'_received_real'],rx,'actual_noise',3e-14)
                dec=project(rx[:,:3]/np.maximum(rx[:,3,None],2**-.5),method)
                close(data[prefix+'_decoded_coordinates'],dec,'independent_projection',3e-14)
                if method=='Q0':aa=(dec+1)*64;ww=np.full((n,3),1/3)
                elif method=='Q1':aa=(dec[:,:2]+1)*64;mw=(dec[:,2]+1)/2;ww=np.column_stack((mw,1-mw))
                else:
                    depths=(dec+1)*64;aa=depths[:,[0,2]];mw=np.divide(depths[:,2]-depths[:,1],depths[:,2]-depths[:,0],out=np.full(n,.5),where=depths[:,2]>depths[:,0]);ww=np.column_stack((mw,1-mw))
                close(data[prefix+'_atoms'],aa,'received_atoms')
                close(data[prefix+'_weights'],ww,'received_weights',3e-14)
                assert (ww>=0).all() and np.max(np.abs(ww.sum(-1)-1))<1e-14
                full=cdf_distance(p,z,ww[v],aa[v]);channel=cdf_distance(source_w,source_a,ww[v],aa[v])
                close(data[prefix+'_full_W1'][v],full,'full_CDF')
                close(data[prefix+'_channel_W1'][v],channel,'channel_CDF')
                assert np.isnan(data[prefix+'_full_W1'][~v]).all() and np.isnan(data[prefix+'_channel_W1'][~v]).all()
                assert (full<=base+channel+4e-11).all()
                norm=np.linalg.norm(rx-tx,axis=-1)
                assert np.max(np.linalg.norm(dec-q,axis=-1)-2*np.sqrt(2)*norm)<4e-14
                factor={'Q0':128*np.sqrt(2/3),'Q1':256,'Q2':128*np.sqrt(10)}[method]
                assert np.max(channel-factor*norm[v])<4e-11
                if condition=='identity':assert channel.max()<4e-11 and np.max(np.abs(dec-q))<4e-14
                row=entry['methods'][method][condition]
                assert row['complex_uses']==n*2 and abs(row['total_energy']-(tx*tx).sum())<2e-8 and abs(row['mean_Es']-1)<1e-13
                for name,x in [('full',full),('channel',channel)]:
                    for k,value in summary(x).items():assert abs(row[name][k]-value)<4e-11
                pooled.setdefault(prefix,[]).append(full)
                uses+=n*2
        difference=data['awgn10_Q2_full_W1'][v]-data['awgn10_Q1_full_W1'][v]
        for k,value in summary(difference).items():assert abs(entry['Q2_minus_Q1_AWGN10'][k]-value)<4e-11
        assert abs(entry['Q2_AWGN10_better_fraction']-float((difference<0).mean()))<1e-14
        grid+=n;validtotal+=int(v.sum())
    assert grid==report['grid_cells']==99840 and validtotal==report['valid_cells']
    assert uses==report['attempted_complex_uses']==1198080
    suffix='native' if args.native else 'local'
    result=dict(state='passed_all_native_Q2_cells_independent_spatial_CDF_and_direct_partition_replay',
        checked_unix=time.time(),verifier_sha256=sha(__file__),artifacts_sha256=artifacts,
        grid_cells=grid,valid_cells=validtotal,fit_candidate_records=fitcandidates,
        method_condition_cells=grid*6,attempted_complex_uses=uses,source_provenance=native,
        maximum_errors=errors,pooled={k:summary(np.concatenate(v)) for k,v in pooled.items()},
        limitation='Four cached training views/uncalibrated cosine sender posterior; no native depth teacher, calibrated truth, task/AP or novelty claim')
    out=ROOT/f'data/provenance/depth-particles-native-Q2-{suffix}-verification-001.json'
    assert not out.exists();out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('state','valid_cells','fit_candidate_records','pooled')}),flush=True)

if __name__=='__main__':main()
