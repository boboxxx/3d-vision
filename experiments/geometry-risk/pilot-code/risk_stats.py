"""Fixed training-only ridge, common support, frame-level scoring and resampling."""
import math
import numpy as np
from scipy.stats import spearmanr

MODELS={'energy':(0,), 'geometry':(1,2,3), 'energy_geometry':(0,1,2,3), 'gradient':(4,), 'shuffled_geometry':(1,2,3)}


def common_rows(report):
    features=[];targets=[];group_ids=[];excluded=[]
    assert len(report['scores'])==len(report['damage'])==32
    for j,(score,damage) in enumerate(zip(report['scores'],report['damage'])):
        assert score['group']==damage['group']==j and len(damage['draw_damage'])==4
        signed=float(np.mean(damage['draw_damage']));variance=float(np.var(damage['draw_damage'],ddof=1))
        assert signed==damage['signed_damage_mean'] and max(signed,0.)==damage['positive_damage_mean'] and variance==damage['sample_variance']
        energy=score['code_energy_mean'];valid=score['valid_count'];candidate=score['candidate_count']
        values=[energy,score['entropy'],score['depth_variance'],score['squared_gradient_sum']]
        if valid<=0 or candidate<=0 or any(v is None or not math.isfinite(v) for v in values) or energy<=0:
            excluded.append(j);continue
        features.append([math.log(energy),score['entropy'],score['depth_variance'],valid/candidate,score['squared_gradient_sum']])
        targets.append([max(signed,0.),signed,variance]);group_ids.append(j)
    return dict(frame_id=report['frame_id'],groups=group_ids,excluded_groups=excluded,
                X=np.asarray(features,np.float64).reshape(-1,5),Y=np.asarray(targets,np.float64).reshape(-1,3))


def shuffle_rows(frames):
    rng=np.random.Generator(np.random.PCG64(2802));out=[];permutations=[]
    for frame in frames:
        perm=rng.permutation(len(frame['X']));x=frame['X'].copy();x[:,1:4]=x[perm,1:4]
        out.append(x);permutations.append(perm.tolist())
    return out,permutations


def ridge_fit(x,y):
    assert x.ndim==2 and x.shape[0]==len(y)>0 and np.isfinite(x).all() and np.isfinite(y).all()
    mean=x.mean(0);scale=x.std(0,ddof=0);scale=np.where(scale==0,1.,scale)
    design=np.column_stack([np.ones(len(x)),(x-mean)/scale])
    penalty=np.diag([0.]+[1.]*x.shape[1])
    coef=np.linalg.solve(design.T@design+penalty,design.T@y)
    return dict(mean=mean,scale=scale,coefficients=coef)


def ridge_predict(model,x):
    return np.column_stack([np.ones(len(x)),(x-model['mean'])/model['scale']])@model['coefficients']


def frame_metrics(pred,y):
    assert len(pred)==len(y)
    if not len(pred):return dict(MSE=None,signed_Spearman=None,positive_Spearman=None,variance_Spearman=None)
    out={'MSE':float(np.mean((pred-y[:,0])**2))}
    for name,column in [('signed',1),('positive',0),('variance',2)]:
        target=y[:,column]
        if len(pred)<3 or np.ptp(pred)==0 or np.ptp(target)==0:value=None
        else:
            value=float(spearmanr(pred,target).statistic)
            if not math.isfinite(value):value=None
        out[name+'_Spearman']=value
    return out


def bootstrap_indices():return np.random.Generator(np.random.PCG64(2803)).integers(0,32,size=(1000,32),dtype=np.int64)


def summarize_frame_values(values,indices):
    assert len(values)==32 and indices.shape==(1000,32)
    a=np.asarray([np.nan if v is None else v for v in values],np.float64)
    valid=np.isfinite(a);counts=valid[indices].sum(1)
    sums=np.where(valid[indices],a[indices],0.).sum(1)
    estimates=np.divide(sums,counts,out=np.full(1000,np.nan),where=counts>0)
    samples=estimates[np.isfinite(estimates)]
    return dict(mean=float(a[valid].mean()) if valid.any() else None,defined_frames=int(valid.sum()),
                undefined_frames=int((~valid).sum()),bootstrap_defined_resamples=len(samples),
                percentile_95_interval=np.percentile(samples,[2.5,97.5]).tolist() if len(samples) else None)


def fit_compare(frames):
    assert len(frames)==64
    shuffled,permutations=shuffle_rows(frames);indices=bootstrap_indices();out={}
    trainy=np.concatenate([f['Y'][:,0] for f in frames[:32]])
    if not len(trainy):return {'state':'no_common_training_support','models':{},'permutations':permutations},indices
    for name,columns in MODELS.items():
        matrices=shuffled if name=='shuffled_geometry' else [f['X'] for f in frames]
        x=np.concatenate([a[:,columns] for a in matrices[:32]],axis=0)
        model=ridge_fit(x,trainy);perframe=[]
        for frame,matrix in zip(frames[32:],matrices[32:]):
            pred=ridge_predict(model,matrix[:,columns]);metrics=frame_metrics(pred,frame['Y'])
            perframe.append(dict(frame_id=frame['frame_id'],groups=frame['groups'],predictions=pred.tolist(),**metrics))
        summaries={metric:summarize_frame_values([f[metric] for f in perframe],indices) for metric in ('MSE','signed_Spearman','positive_Spearman','variance_Spearman')}
        out[name]=dict(columns=list(columns),fit={k:v.tolist() for k,v in model.items()},fit_rows=len(trainy),per_frame=perframe,summary=summaries)
    return dict(state='completed_prelocked_five_ridge_comparisons',models=out,permutations=permutations,
                common_support=[dict(frame_id=f['frame_id'],groups=f['groups'],excluded_groups=f['excluded_groups']) for f in frames]),indices
