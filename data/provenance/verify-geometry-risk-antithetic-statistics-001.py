"""Independently recompute frozen predictor results from raw paired losses."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[2];PREFIX='geometry-risk-antithetic-001'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def same(a,b):
    assert (a is None)==(b is None)
    if b is not None:np.testing.assert_allclose(a,b,rtol=2e-11,atol=1e-14)
def rank(a,b):
    if len(a)<3 or np.ptp(a)==0 or np.ptp(b)==0:return None
    return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--transferred-records',type=Path);p.add_argument('--transferred-bootstrap',type=Path)
    args=p.parse_args();assert not args.output.exists() and (args.transferred_records is None)==(args.transferred_bootstrap is None)
    rp=ROOT/'data/analysis'/f'{PREFIX}-frozen-predictors-001.json';r=read(rp)
    mp=ROOT/'data/runs'/f'{PREFIX}.json';run=read(mp);ap=ROOT/'data/provenance'/f'{PREFIX}-native-audit.json';audit=read(ap)
    op=ROOT/'data/analysis/geometry-risk-pilot-001-ridge-001.json';old=read(op)
    assert r['state']=='completed_frozen_five_predictor_fresh8_diagnosis' and audit['state']=='passed_full8_independent_antithetic_native_audit'
    assert sha(mp)==r['manifest_sha256']==audit['manifest_sha256'] and sha(ap)==r['native_audit_sha256']
    assert sha(op)==r['frozen_predictors_sha256']==run['eligibility']['artifacts_sha256'][str(op.relative_to(ROOT))]
    for path,h in r['source_sha256'].items():assert sha(ROOT/path)==h
    frames=[]
    for fid in run['ids']:
        path=Path(run['frames'][fid]['report_path'])
        if args.transferred_records is not None:path=args.transferred_records/path.name
        assert sha(path)==r['report_sha256'][fid]==audit['summaries'][fid]['frame_report_sha256']
        report=read(path);x=[];y=[];groups=[];excluded=[]
        for j in range(32):
            score=report['scores'][j];damage=report['damage'][j]
            plus=np.asarray(damage['positive_loss'],np.float64)-report['clean_loss']
            minus=np.asarray(damage['negative_loss'],np.float64)-report['clean_loss']
            q=(plus+minus)/2;s=(plus-minus)/2;mean=float(q.mean());var=float(s.var(ddof=1))
            assert damage['signed_symmetric_mean']==mean and damage['positive_symmetric_mean']==max(mean,0.) and damage['antisymmetric_sample_variance']==var
            values=[score['code_energy_mean'],score['entropy'],score['depth_variance'],score['squared_gradient_sum']]
            if score['valid_count']<=0 or score['candidate_count']<=0 or any(v is None or not math.isfinite(v) for v in values) or values[0]<=0:
                excluded.append(j);continue
            x.append([math.log(values[0]),values[1],values[2],score['valid_count']/score['candidate_count'],values[3]])
            y.append([mean,max(mean,0.),var]);groups.append(j)
        frames.append(dict(frame_id=fid,groups=groups,excluded_groups=excluded,X=np.asarray(x,np.float64).reshape(-1,5),Y=np.asarray(y,np.float64).reshape(-1,3)))
    assert len(frames)==8 and r['common_support']==[{k:f[k] for k in ('frame_id','groups','excluded_groups')} for f in frames]
    rng=np.random.Generator(np.random.PCG64(2802))
    for support,permutation in zip(old['common_support'],old['permutations']):assert rng.permutation(len(support['groups'])).tolist()==permutation
    shuffled=[];permutations=[]
    for frame in frames:
        perm=rng.permutation(len(frame['groups']));x=frame['X'].copy();x[:,1:4]=frame['X'][perm,1:4];shuffled.append(x);permutations.append(perm.tolist())
    assert permutations==r['permutations']
    ip=Path(r['bootstrap_indices_path']) if args.transferred_bootstrap is None else args.transferred_bootstrap
    assert sha(ip)==r['bootstrap_indices_sha256'];indices=np.load(ip,allow_pickle=False)
    np.testing.assert_array_equal(indices,np.random.Generator(np.random.PCG64(2805)).integers(0,8,(1000,8),dtype=np.int64))
    metrics=['signed_symmetric_Spearman','positive_symmetric_Spearman','antisymmetric_variance_Spearman'];checks={}
    assert set(r['models'])==set(old['models']) and len(r['models'])==5
    for name,model in r['models'].items():
        original=old['models'][name];fit=original['fit'];cols=original['columns']
        assert model['frozen_fit']==fit and model['columns']==cols
        matrices=shuffled if name=='shuffled_geometry' else [f['X'] for f in frames];values={k:[] for k in metrics}
        for f,x,row in zip(frames,matrices,model['per_frame']):
            pred=np.asarray([fit['coefficients'][0]+math.fsum(fit['coefficients'][i+1]*(a[col]-fit['mean'][i])/fit['scale'][i] for i,col in enumerate(cols)) for a in x])
            assert row['frame_id']==f['frame_id'] and row['groups']==f['groups']
            np.testing.assert_allclose(row['predictions'],pred,rtol=2e-11,atol=1e-13)
            for j,key in enumerate(metrics):
                value=rank(pred,f['Y'][:,j]);same(row[key],value);values[key].append(value)
        summaries={}
        for key,v in values.items():
            defined=[a for a in v if a is not None];boot=[]
            for sample in indices:
                b=[v[i] for i in sample if v[i] is not None]
                if b:boot.append(float(np.mean(b)))
            summary=dict(mean=float(np.mean(defined)) if defined else None,defined_frames=len(defined),undefined_frames=8-len(defined),
                         bootstrap_defined_resamples=len(boot),percentile_95_interval=np.percentile(boot,[2.5,97.5]).tolist() if boot else None)
            actual=model['summary'][key]
            for field in ('defined_frames','undefined_frames','bootstrap_defined_resamples'):assert actual[field]==summary[field]
            same(actual['mean'],summary['mean']);same(actual['percentile_95_interval'],summary['percentile_95_interval']);summaries[key]=summary
        checks[name]=summaries
    output=dict(state='passed_independent_fresh8_frozen_predictor_statistics',verifier_sha256=sha(Path(__file__)),result_sha256=sha(rp),
                manifest_sha256=sha(mp),native_audit_sha256=sha(ap),frozen_predictors_sha256=sha(op),model_checks=checks,
                transferred_inputs=args.transferred_records is not None,scope='Independent raw paired targets frozen coefficient predictions ranks and eight-frame bootstrap; not AP/allocation')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(output,f,indent=2,allow_nan=False)
    print(json.dumps({'state':output['state'],'models':len(checks)}))

if __name__=='__main__':main()
