"""Analytic paired decomposition, complete chronology, receiver path and fixed predictor gates."""
import argparse,copy,hashlib,json,os,tempfile
from pathlib import Path
import numpy as np
import torch
from pair_noise import PairNoise,pair_statistics
from channel import AntitheticChannel
from analyze import evaluate,summarize
ROOT=Path(__file__).resolve().parents[3]

def reject(fn):
    try:fn()
    except (ValueError,RuntimeError):return
    raise AssertionError('invalid chronology or overwrite was accepted')

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);out=Path(p.parse_args().output).resolve();assert not out.exists()
    assert not torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    checks=[]
    e=np.arange(48,dtype=float).reshape(16,3)/32-.7;g=np.array([2.,-3.,.5]);h=np.array([[2.,1.,0.],[1.,-4.,.2],[0.,.2,1.]])
    q=.5*np.einsum('ni,ij,nj->n',e,h,e);s=e@g;value=pair_statistics(3+q+s,3+q-s,3,s)
    np.testing.assert_allclose(value['symmetric'],q,rtol=1e-13,atol=1e-14);np.testing.assert_allclose(value['antisymmetric'],s,rtol=1e-13,atol=1e-14)
    checks.append('analytic_quadratic_signed_curvature_and_linear_odd_part')
    odd=e[:,0]**3;v=pair_statistics(odd,-odd,0,np.zeros(16));assert v['signed_symmetric_mean']==0
    np.testing.assert_array_equal(v['antisymmetric_minus_linear'],odd)
    negative=pair_statistics(-np.ones(16),-np.ones(16),0,np.zeros(16));assert negative['signed_symmetric_mean']==-1 and negative['positive_symmetric_mean']==0
    checks.append('nonlinear_odd_residual_and_unclipped_negative_symmetric_mean')
    groups=np.arange(32);draw=PairNoise(groups);reference=np.random.Generator(np.random.PCG64(2804));before_global=copy.deepcopy(np.random.get_state())
    draw.sample('test','clean');count=1
    for group in range(32):
        for pair in range(16):
            positive,record=draw.sample('test','group',group,pair,1)
            ref=reference.standard_normal((32,2))*np.sqrt(.05);ref[groups!=group]=0;ref=ref.astype(np.float32)
            np.testing.assert_array_equal(positive,ref);assert record['after']==reference.bit_generator.state
            negative,record=draw.sample('test','group',group,pair,-1)
            assert negative.tobytes()==(-positive).tobytes() and record['before']==record['after'] and not record['independent_draw'];count+=2
    actual,record=draw.sample('test','full_awgn');ref=(reference.standard_normal((32,2))*np.sqrt(.05)).astype(np.float32)
    np.testing.assert_array_equal(actual,ref);assert draw.draws==513 and count+1==1026 and draw.index==0
    after_global=np.random.get_state();assert before_global[0]==after_global[0] and before_global[2:]==after_global[2:];np.testing.assert_array_equal(before_global[1],after_global[1])
    checks.append('complete_1026_attempt_noise_replay_and_rng_isolation')
    reject(lambda:draw.sample('test','clean'));bad=PairNoise(groups);reject(lambda:bad.sample('a','group',0,0,-1))
    bad.sample('a','clean');state=copy.deepcopy(bad.rng.bit_generator.state);reject(lambda:bad.sample('a','group',1,0,1));assert state==bad.rng.bit_generator.state
    bad.sample('a','group',0,0,1);reject(lambda:bad.sample('a','full_awgn'));reject(lambda:bad.sample('b','group',0,0,-1))
    checks.append('wrong_order_frame_reuse_and_missing_negative_refused')
    class Identity(torch.nn.Module):
        kind='identity'
        def forward(self,x,snr,generator=None):return x,dict(total_complex_uses=x.shape[1])
    link=torch.nn.Module();link.channel=Identity();link.eval()
    with tempfile.TemporaryDirectory() as directory:
        c=AntitheticChannel(link,directory,[(1,2,20,1248),(1,2,20,1248),(1,8,20,156)])
        try:
            existing=Path(directory)/'saved-clean.npy';existing.write_bytes(b'preserved');reject(lambda:c.prepare('saved','clean'));assert existing.read_bytes()==b'preserved' and c.prepared==0
            x=torch.ones((1,62400,2),dtype=torch.float32);x[0,0,0]=-0.
            c.begin_frame();c.prepare('fresh','clean');received,account=link.channel(x,10.)
            assert received.detach().numpy().tobytes()==x.numpy().tobytes()
            grad,=torch.autograd.grad(received.square().sum(),c.leaf);torch.testing.assert_close(grad,2*x)
            c.finish();c.prepare('fresh','group',0,0,1);plus=c.noise.copy();received,_=link.channel(x,10.);torch.testing.assert_close(received.detach(),x+torch.from_numpy(plus)[None],rtol=0,atol=0);c.finish()
            c.prepare('fresh','group',0,0,-1);assert c.noise.tobytes()==(-plus).tobytes();link.channel(x,10.);c.finish();assert c.calls==c.prepared==3
        finally:c.close()
    checks.append('actual_symbol_leaf_gradient_identity_opposite_noise_and_file_preservation')
    old=json.loads((ROOT/'data/analysis/geometry-risk-pilot-001-ridge-001.json').read_text());saved=json.dumps(old,sort_keys=True)
    frames=[dict(frame_id=str(i),groups=[0,1,2],excluded_groups=list(range(3,32)),X=np.arange(15,dtype=float).reshape(3,5)+i,Y=np.arange(9,dtype=float).reshape(3,3)) for i in range(8)]
    result,indices=evaluate(old,frames);assert json.dumps(old,sort_keys=True)==saved and len(result['models'])==5
    np.testing.assert_array_equal(indices,np.random.Generator(np.random.PCG64(2805)).integers(0,8,(1000,8),dtype=np.int64))
    for name,model in result['models'].items():assert model['frozen_fit']==old['models'][name]['fit']
    assert summarize([None]*8,indices)['mean'] is None and summarize([2.]*8,indices)['percentile_95_interval']==[2.,2.]
    checks.append('frozen_predictors_unchanged_and_eight_frame_bootstrap')
    result=dict(state='passed',tests=len(checks),checks=checks,CUDA_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],torch=torch.__version__,numpy=np.__version__,
                source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))})
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'state':'passed','tests':len(checks)}))

if __name__=='__main__':main()
