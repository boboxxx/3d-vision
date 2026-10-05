"""Frozen old predictors evaluated only after full fresh native audit; no refitting."""
import json,hashlib,math,time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[3];PREFIX='geometry-risk-antithetic-001'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())

def common_frame(report):
    x=[];y=[];groups=[];excluded=[]
    for j,(score,damage) in enumerate(zip(report['scores'],report['damage'])):
        assert score['group']==damage['group']==j
        values=[score['code_energy_mean'],score['entropy'],score['depth_variance'],score['squared_gradient_sum']]
        if score['valid_count']<=0 or score['candidate_count']<=0 or any(v is None or not math.isfinite(v) for v in values) or values[0]<=0:
            excluded.append(j);continue
        x.append([math.log(values[0]),values[1],values[2],score['valid_count']/score['candidate_count'],values[3]])
        y.append([damage['signed_symmetric_mean'],damage['positive_symmetric_mean'],damage['antisymmetric_sample_variance']]);groups.append(j)
    return dict(frame_id=report['frame_id'],groups=groups,excluded_groups=excluded,X=np.asarray(x,np.float64).reshape(-1,5),Y=np.asarray(y,np.float64).reshape(-1,3))

def rank(a,b):
    if len(a)<3 or np.ptp(a)==0 or np.ptp(b)==0:return None
    value=float(spearmanr(a,b).statistic)
    return value if math.isfinite(value) else None

def summarize(values,indices):
    assert len(values)==8 and indices.shape==(1000,8)
    a=np.asarray([np.nan if v is None else v for v in values]);valid=np.isfinite(a)
    counts=valid[indices].sum(1);sums=np.where(valid[indices],a[indices],0.).sum(1)
    boot=np.divide(sums,counts,out=np.full(1000,np.nan),where=counts>0);boot=boot[np.isfinite(boot)]
    return dict(mean=float(a[valid].mean()) if valid.any() else None,defined_frames=int(valid.sum()),undefined_frames=int((~valid).sum()),
                bootstrap_defined_resamples=len(boot),percentile_95_interval=np.percentile(boot,[2.5,97.5]).tolist() if len(boot) else None)

def evaluate(old,frames):
    assert len(frames)==8 and set(old['models'])=={'energy','geometry','energy_geometry','gradient','shuffled_geometry'}
    rng=np.random.Generator(np.random.PCG64(2802))
    for support,permutation in zip(old['common_support'],old['permutations']):
        assert rng.permutation(len(support['groups'])).tolist()==permutation
    shuffled=[];permutations=[]
    for frame in frames:
        perm=rng.permutation(len(frame['groups']));x=frame['X'].copy();x[:,1:4]=x[perm,1:4]
        shuffled.append(x);permutations.append(perm.tolist())
    indices=np.random.Generator(np.random.PCG64(2805)).integers(0,8,(1000,8),dtype=np.int64)
    models={};metrics=['signed_symmetric_Spearman','positive_symmetric_Spearman','antisymmetric_variance_Spearman']
    for name,original in old['models'].items():
        fit=original['fit'];columns=original['columns'];rows=[]
        matrices=shuffled if name=='shuffled_geometry' else [f['X'] for f in frames]
        for frame,x in zip(frames,matrices):
            pred=np.column_stack([np.ones(len(x)),(x[:,columns]-np.asarray(fit['mean']))/np.asarray(fit['scale'])])@np.asarray(fit['coefficients'])
            rows.append(dict(frame_id=frame['frame_id'],groups=frame['groups'],predictions=pred.tolist(),
                             **{metric:rank(pred,frame['Y'][:,k]) for k,metric in enumerate(metrics)}))
        models[name]=dict(frozen_fit=fit,columns=columns,per_frame=rows,
                          summary={metric:summarize([r[metric] for r in rows],indices) for metric in metrics})
    return dict(state='completed_frozen_five_predictor_fresh8_diagnosis',models=models,permutations=permutations,
                common_support=[{k:f[k] for k in ('frame_id','groups','excluded_groups')} for f in frames]),indices

def main():
    runpath=ROOT/'data/runs'/f'{PREFIX}.json';run=read(runpath)
    ap=ROOT/'data/provenance'/f'{PREFIX}-native-audit.json';a=read(ap)
    assert run['state']=='finished8_paired_observations_independent_audit_pending'
    assert a['state']=='passed_full8_independent_antithetic_native_audit' and a['manifest_sha256']==sha(runpath) and a['native_rows']==8208
    assert a['actual_terminal_PID']==run['pid']
    oldpath=ROOT/'data/analysis/geometry-risk-pilot-001-ridge-001.json';old=read(oldpath)
    assert sha(oldpath)==run['eligibility']['artifacts_sha256'][str(oldpath.relative_to(ROOT))]
    reports=[];hashes={}
    for frame in run['ids']:
        p=Path(run['frames'][frame]['report_path']);assert sha(p)==run['frames'][frame]['report_sha256']==a['summaries'][frame]['frame_report_sha256']
        reports.append(common_frame(read(p)));hashes[frame]=sha(p)
    result,indices=evaluate(old,reports)
    out=ROOT/'data/analysis'/f'{PREFIX}-frozen-predictors-001.json';ip=ROOT/'data/analysis'/f'{PREFIX}-bootstrap-001.npy'
    assert not out.exists() and not ip.exists();out.parent.mkdir(parents=True,exist_ok=True)
    np.save(ip,indices,allow_pickle=False)
    result.update(manifest_sha256=sha(runpath),native_audit_sha256=sha(ap),frozen_predictors_sha256=sha(oldpath),
                  report_sha256=hashes,bootstrap_indices_path=str(ip),bootstrap_indices_sha256=sha(ip),finished_unix=time.time(),
                  source_sha256={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')},
                  scope='Exploratory training-scene mechanism diagnosis; no new fit, allocation, AP or inference-only gradient availability')
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v['summary'] for k,v in result['models'].items()}))

if __name__=='__main__':main()
