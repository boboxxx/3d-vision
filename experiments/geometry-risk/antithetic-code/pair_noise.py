"""Strict fixed-roster paired noise; no model or target inputs."""
import copy
import math
import numpy as np


class PairNoise:
    def __init__(self, group_ids):
        self.groups=np.asarray(group_ids,dtype=np.int64)
        assert self.groups.ndim==1 and set(self.groups)==set(range(32))
        self.rng=np.random.Generator(np.random.PCG64(2804))
        self.index=0;self.frame=None;self.seen=set();self.positive=None;self.draws=0

    def sample(self,frame,mode,group=None,pair=None,sign=None):
        i=self.index
        expected=('clean',None,None,None) if i==0 else ('full_awgn',None,None,None) if i==1025 else ('group',(i-1)//32,((i-1)//2)%16,1 if i%2 else -1)
        if (mode,group,pair,sign)!=expected or (i and frame!=self.frame) or (i==0 and frame in self.seen):
            raise ValueError('paired noise chronology differs from fixed 1026-attempt frame')
        if i==0:self.frame=frame;self.seen.add(frame)
        before=copy.deepcopy(self.rng.bit_generator.state);count=len(self.groups)
        if mode=='clean':noise=np.zeros((count,2),np.float32)
        elif sign==-1:
            assert self.positive is not None
            noise=-self.positive;self.positive=None
        else:
            noise=self.rng.standard_normal((count,2))*math.sqrt(.05);self.draws+=1
            if mode=='group':noise[self.groups!=group]=0
            noise=noise.astype(np.float32)
            if mode=='group':self.positive=noise.copy()
        after=copy.deepcopy(self.rng.bit_generator.state)
        record=dict(before=before,after=after,draw_shape=[count,2] if mode=='full_awgn' or sign==1 else [0,2],
                    selected_symbols=int((self.groups==group).sum()) if mode=='group' else count if mode=='full_awgn' else 0,
                    attempted_complex_uses=count,local_N0=0. if mode=='clean' else .1,independent_draw=mode=='full_awgn' or sign==1)
        self.index=(i+1)%1026
        return noise,record


def pair_statistics(positive,negative,clean,linear):
    p=np.asarray(positive,np.float64);n=np.asarray(negative,np.float64);l=np.asarray(linear,np.float64)
    assert p.shape==n.shape==l.shape==(16,) and np.isfinite(p).all() and np.isfinite(n).all() and np.isfinite(l).all()
    dp=p-clean;dn=n-clean;q=(dp+dn)/2;s=(dp-dn)/2
    return dict(positive_loss=p.tolist(),negative_loss=n.tolist(),positive_damage=dp.tolist(),negative_damage=dn.tolist(),
                symmetric=q.tolist(),antisymmetric=s.tolist(),linear=l.tolist(),antisymmetric_minus_linear=(s-l).tolist(),
                signed_symmetric_mean=float(q.mean()),positive_symmetric_mean=max(float(q.mean()),0.),
                symmetric_sample_variance=float(q.var(ddof=1)),antisymmetric_mean=float(s.mean()),
                antisymmetric_sample_variance=float(s.var(ddof=1)),linear_sample_variance=float(l.var(ddof=1)))
