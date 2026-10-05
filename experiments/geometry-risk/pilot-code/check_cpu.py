"""Analytic ridge, no-test-leakage, common supports and correlated-cell resampling."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from risk_stats import ridge_fit,ridge_predict,frame_metrics,summarize_frame_values,bootstrap_indices,shuffle_rows,common_rows
ROOT=Path(__file__).resolve().parents[3]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);output=Path(p.parse_args().output).resolve();assert not output.exists()
    x=np.array([[-1.],[0.],[1.]]);y=2+3*x[:,0];model=ridge_fit(x,y)
    np.testing.assert_allclose(ridge_predict(model,x),2+2.25*x[:,0],rtol=1e-14,atol=1e-14)
    np.testing.assert_allclose(ridge_predict(model,np.array([[100.]])),[227.],rtol=1e-14)
    assert model['mean'][0]==0 and model['scale'][0]==np.std(x,ddof=0)
    const=ridge_fit(np.ones((3,2)),np.array([1.,2.,3.]));np.testing.assert_allclose(ridge_predict(const,np.ones((2,2))),[2.,2.])
    assert frame_metrics(np.ones(3),np.ones((3,3)))['signed_Spearman'] is None
    assert frame_metrics(np.array([]),np.empty((0,3)))['MSE'] is None
    metric=frame_metrics(np.array([3.,1.,2.]),np.array([[3.,1.,3.],[1.,3.,1.],[2.,2.,2.]]))
    assert metric['MSE']==0 and metric['positive_Spearman']==1 and metric['signed_Spearman']==-1 and metric['variance_Spearman']==1
    a=bootstrap_indices();assert a.shape==(1000,32) and a.min()==0 and a.max()==31
    np.testing.assert_array_equal(a,np.random.Generator(np.random.PCG64(2803)).integers(0,32,(1000,32),dtype=np.int64))
    summary=summarize_frame_values([2.]*31+[None],a);assert summary['mean']==2 and summary['defined_frames']==31 and summary['percentile_95_interval']==[2.,2.]
    assert summarize_frame_values([None]*32,a)['mean'] is None
    frames=[dict(X=np.arange(25,dtype=float).reshape(5,5)+i*100) for i in range(64)]
    shuffled,perms=shuffle_rows(frames)
    for original,actual,perm in zip(frames,shuffled,perms):
        np.testing.assert_array_equal(actual[:,[0,4]],original['X'][:,[0,4]])
        np.testing.assert_array_equal(actual[:,1:4],original['X'][perm,1:4])
    report=dict(frame_id='synthetic',scores=[],damage=[])
    for j in range(32):
        report['scores'].append(dict(group=j,code_energy_mean=1.,valid_count=0 if j==31 else 1,candidate_count=2,entropy=1.,depth_variance=2.,squared_gradient_sum=3.))
        report['damage'].append(dict(group=j,draw_damage=[-1.,1.,-1.,1.],signed_damage_mean=0.,positive_damage_mean=0.,sample_variance=4/3))
    complete=common_rows(report);assert complete['groups']==list(range(31)) and complete['excluded_groups']==[31]
    assert complete['X'].shape==(31,5) and complete['Y'].shape==(31,3)
    result=dict(state='passed',tests=5,scope='analytic_statistics_CPU_only_no_native_observations',numpy=np.__version__,
                source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))})
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'state':'passed','tests':5}))


if __name__=='__main__':main()
