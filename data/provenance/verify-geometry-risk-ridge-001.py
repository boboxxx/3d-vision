"""Independent auxiliary calibration audit: augmented least squares and explicit frame resampling."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[2];PREFIX='geometry-risk-pilot-001'
MODELS={'energy':[0],'geometry':[1,2,3],'energy_geometry':[0,1,2,3],'gradient':[4],'shuffled_geometry':[1,2,3]}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())


def raw_frame(report):
    x=[];y=[];groups=[];excluded=[]
    for group in range(32):
        s=report['scores'][group];d=report['damage'][group]
        mean=float(np.mean(d['draw_damage']));variance=float(np.var(d['draw_damage'],ddof=1))
        assert d['signed_damage_mean']==mean and d['positive_damage_mean']==max(mean,0.) and d['sample_variance']==variance
        values=[s['code_energy_mean'],s['entropy'],s['depth_variance'],s['squared_gradient_sum']]
        if s['valid_count']<=0 or s['candidate_count']<=0 or any(v is None or not math.isfinite(v) for v in values) or values[0]<=0:
            excluded.append(group);continue
        x.append([math.log(values[0]),values[1],values[2],s['valid_count']/s['candidate_count'],values[3]])
        y.append([max(mean,0.),mean,variance]);groups.append(group)
    return dict(frame_id=report['frame_id'],X=np.asarray(x,dtype=np.float64).reshape(-1,5),Y=np.asarray(y,dtype=np.float64).reshape(-1,3),groups=groups,excluded_groups=excluded)


def correlation(a,b):
    if len(a)<3 or np.ptp(a)==0 or np.ptp(b)==0:return None
    return float(np.corrcoef(rankdata(a,method='average'),rankdata(b,method='average'))[0,1])


def mean_bootstrap(values,indices):
    defined=[v for v in values if v is not None];boot=[]
    for row in indices:
        sample=[values[i] for i in row if values[i] is not None]
        if sample:boot.append(float(np.mean(sample)))
    return dict(mean=float(np.mean(defined)) if defined else None,defined_frames=len(defined),undefined_frames=32-len(defined),
                bootstrap_defined_resamples=len(boot),percentile_95_interval=np.percentile(boot,[2.5,97.5]).tolist() if boot else None)


def compare(actual,expected):
    assert (actual is None)==(expected is None)
    if expected is not None:np.testing.assert_allclose(actual,expected,rtol=2e-11,atol=1e-15)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--transferred-records',type=Path)
    p.add_argument('--transferred-bootstrap',type=Path)
    args=p.parse_args();assert not args.output.exists()
    assert (args.transferred_records is None)==(args.transferred_bootstrap is None)
    resultpath=ROOT/'data/analysis'/f'{PREFIX}-ridge-001.json';result=read(resultpath)
    runpath=ROOT/'data/runs'/f'{PREFIX}.json';run=read(runpath)
    auditpath=ROOT/'data/provenance'/f'{PREFIX}-native-audit.json';audit=read(auditpath)
    assert audit['state']=='passed_full64_independent_native_audit' and audit['manifest_sha256']==sha(runpath)==result['manifest_sha256']
    assert sha(auditpath)==result['native_audit_sha256'] and sha(ROOT/'experiments/geometry-risk/pilot-execution-001.md')==result['protocol_sha256']
    for path,value in result['analysis_sources'].items():assert sha(ROOT/path)==value
    ids=run['fit_ids']+run['out_of_fit_ids'];assert len(ids)==64 and len(run['fit_ids'])==32
    frames=[]
    for frame in ids:
        report=Path(run['frames'][frame]['report_path'])
        if args.transferred_records is not None:report=args.transferred_records/report.name
        assert sha(report)==result['report_sha256'][frame]==audit['summaries'][frame]['frame_report_sha256']
        frames.append(raw_frame(read(report)))
    shuffle_rng=np.random.Generator(np.random.PCG64(2802));shuffled=[];perms=[]
    for frame in frames:
        perm=shuffle_rng.permutation(len(frame['X']));x=frame['X'].copy();x[:,1:4]=frame['X'][perm,1:4]
        shuffled.append(x);perms.append(perm.tolist())
    assert perms==result['permutations']
    indicespath=Path(result['bootstrap_indices_path'])
    if args.transferred_bootstrap is not None:indicespath=args.transferred_bootstrap
    assert sha(indicespath)==result['bootstrap_indices_sha256']
    indices=np.load(indicespath,allow_pickle=False)
    np.testing.assert_array_equal(indices,np.random.Generator(np.random.PCG64(2803)).integers(0,32,(1000,32),dtype=np.int64))
    trainy=np.concatenate([f['Y'][:,0] for f in frames[:32]])
    if len(trainy)==0:
        assert result['state']=='no_common_training_support' and not result['models']
        model_checks={}
    else:
        assert result['state']=='completed_prelocked_five_ridge_comparisons' and set(result['models'])==set(MODELS)
        assert result['common_support']==[dict(frame_id=f['frame_id'],groups=f['groups'],excluded_groups=f['excluded_groups']) for f in frames]
        model_checks={}
        for name,columns in MODELS.items():
            actual=result['models'][name];matrices=shuffled if name=='shuffled_geometry' else [f['X'] for f in frames]
            x=np.concatenate([a[:,columns] for a in matrices[:32]])
            mean=x.mean(0);scale=x.std(0,ddof=0);scale[scale==0]=1.
            design=np.column_stack([np.ones(len(x)),(x-mean)/scale])
            # Solve a DIFFERENT algebraic system: augmented least squares, no penalized intercept.
            regularizer=np.column_stack([np.zeros(len(columns)),np.eye(len(columns))])
            augmented=np.vstack([design,regularizer]);rhs=np.concatenate([trainy,np.zeros(len(columns))])
            coefficients=np.linalg.lstsq(augmented,rhs,rcond=None)[0]
            assert actual['columns']==columns and actual['fit_rows']==len(trainy)
            np.testing.assert_allclose(actual['fit']['mean'],mean,rtol=1e-13,atol=1e-13)
            np.testing.assert_allclose(actual['fit']['scale'],scale,rtol=1e-13,atol=1e-13)
            np.testing.assert_allclose(actual['fit']['coefficients'],coefficients,rtol=2e-11,atol=1e-13)
            perframe=[]
            for frame,matrix,row in zip(frames[32:],matrices[32:],actual['per_frame']):
                prediction=np.column_stack([np.ones(len(matrix)),(matrix[:,columns]-mean)/scale])@coefficients
                assert row['frame_id']==frame['frame_id'] and row['groups']==frame['groups']
                np.testing.assert_allclose(row['predictions'],prediction,rtol=2e-11,atol=1e-13)
                metrics=dict(MSE=float(np.mean((prediction-frame['Y'][:,0])**2)) if len(prediction) else None,
                             signed_Spearman=correlation(prediction,frame['Y'][:,1]),positive_Spearman=correlation(prediction,frame['Y'][:,0]),
                             variance_Spearman=correlation(prediction,frame['Y'][:,2]))
                for metric,value in metrics.items():compare(row[metric],value)
                perframe.append(metrics)
            summaries={}
            for metric in ('MSE','signed_Spearman','positive_Spearman','variance_Spearman'):
                expected=mean_bootstrap([f[metric] for f in perframe],indices);a=actual['summary'][metric]
                for field in ('defined_frames','undefined_frames','bootstrap_defined_resamples'):assert a[field]==expected[field]
                compare(a['mean'],expected['mean'])
                assert (a['percentile_95_interval'] is None)==(expected['percentile_95_interval'] is None)
                if expected['percentile_95_interval'] is not None:np.testing.assert_allclose(a['percentile_95_interval'],expected['percentile_95_interval'],rtol=2e-11,atol=1e-15)
                summaries[metric]=expected
            model_checks[name]=dict(fit_rows=len(trainy),out_frames=32,independent_summaries=summaries)
    output=dict(state='passed_independent_fixed_ridge_and_frame_bootstrap_audit',result_sha256=sha(resultpath),native_audit_sha256=sha(auditpath),
                verifier_sha256=sha(Path(__file__)),model_checks=model_checks,
                transferred_inputs=args.transferred_records is not None,
                scope='Independent raw-report features targets augmented least squares ranks and frame resampling. Not perception generalization or inference-only gradient availability.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(output,f,indent=2,allow_nan=False)
    print(json.dumps({'state':output['state'],'models':len(model_checks)}))


if __name__=='__main__':main()
