"""Two ordered bounded depths and an independently communicated modal mass."""
import numpy as np
import torch

def quantile(probability, edges, levels):
    p = np.asarray(probability, dtype=np.float64); p = p / p.sum()
    assert p.ndim == 1 and (p >= 0).all() and np.isfinite(p).all()
    cumulative = p.cumsum(); cumulative[-1] = 1.
    levels = np.asarray(levels, dtype=np.float64)
    assert ((levels >= 0) & (levels <= 1)).all()
    index = np.searchsorted(cumulative, levels, side='left').clip(0,len(p)-1)
    nonzero = np.flatnonzero(p > 0)
    index = np.where(levels == 0, nonzero[0], index)
    index = np.where(levels == 1, nonzero[-1], index)
    assert (p[index] > 0).all()
    previous = np.r_[0., cumulative[:-1]][index]
    return edges[index] + (levels-previous) / p[index] * np.diff(edges)[index]

def histogram_W1(probability, edges, atoms, first_mass):
    p = np.asarray(probability,dtype=np.float64); edges=np.asarray(edges,dtype=np.float64)
    atoms=np.asarray(atoms,dtype=np.float64); mass=np.asarray(first_mass,dtype=np.float64)
    assert p.shape[-1]+1==len(edges) and atoms.shape[-1]==2
    assert np.isfinite(p).all() and (p>=0).all() and (p.sum(-1)>0).all()
    assert np.isfinite(atoms).all() and (np.diff(atoms,axis=-1)>=0).all()
    assert np.isfinite(mass).all() and ((mass>=0)&(mass<=1)).all()
    p=p/p.sum(-1,keepdims=True)
    end=p.cumsum(-1)[..., :,None]; start=end-p[..., :,None]
    lower=np.stack((np.zeros_like(mass),mass),-1)[...,None,:]
    upper=np.stack((mass,np.ones_like(mass)),-1)[...,None,:]
    lo=np.maximum(start,lower);hi=np.minimum(end,upper);width=np.maximum(hi-lo,0)
    divisor=np.where(p>0,p,1)[..., :,None]
    first=edges[:-1,None]+(lo-start)/divisor*np.diff(edges)[:,None]-atoms[...,None,:]
    last=edges[:-1,None]+(hi-start)/divisor*np.diff(edges)[:,None]-atoms[...,None,:]
    total=np.abs(first)+np.abs(last)
    area=width*np.where(first*last>=0,total/2,(first**2+last**2)/(2*np.where(total>0,total,1)))
    return area.sum(axis=(-2,-1))

def fit_two(probability,edges):
    """All piecewise-affine derivative regions, not a local optimizer/grid."""
    p=np.asarray(probability,dtype=np.float64);p=p/p.sum();cdf=p.cumsum();cdf[-1]=1.
    boundaries=np.unique(np.clip(np.r_[0.,1.,cdf,2*cdf,2*cdf-1],0,1))
    candidates=list(boundaries)
    def derivative(m):
        return 2*quantile(p,edges,m)-quantile(p,edges,m/2)-quantile(p,edges,(1+m)/2)
    for lo,hi in zip(boundaries[:-1],boundaries[1:]):
        if hi-lo<1e-14:continue
        x1=lo+(hi-lo)/4;x2=lo+3*(hi-lo)/4
        y1=float(derivative(x1));y2=float(derivative(x2));slope=(y2-y1)/(x2-x1)
        if abs(slope)>1e-12:
            root=x1-y1/slope
            if lo<root<hi:candidates.append(root)
    candidates=np.unique(candidates)
    atoms=np.stack((quantile(p,edges,candidates/2),quantile(p,edges,(1+candidates)/2)),-1)
    values=histogram_W1(p[None,:],edges,atoms,candidates)
    minimum=float(values.min());eligible=np.flatnonzero(values<=minimum+1e-11)
    selected=int(eligible[0])
    return dict(atoms=atoms[selected],mass=float(candidates[selected]),W1=float(values[selected]),
                candidate_splits=candidates,candidate_objectives=values,tied_candidates=len(eligible))

def law_W1(a,m,b,n):
    a=np.asarray(a);b=np.asarray(b);m=np.asarray(m);n=np.asarray(n)
    return (np.minimum(m,n)*np.abs(a[...,0]-b[...,0])
        +np.maximum(m-n,0)*np.abs(a[...,0]-b[...,1])
        +np.maximum(n-m,0)*np.abs(a[...,1]-b[...,0])
        +(1-np.maximum(m,n))*np.abs(a[...,1]-b[...,1]))

def source_W1(p,q,edges):
    delta=np.asarray(p)/np.sum(p)-np.asarray(q)/np.sum(q)
    last=delta.cumsum();first=last-delta;total=np.abs(first)+np.abs(last)
    return float(np.sum(np.diff(edges)*np.where(first*last>=0,total/2,
        (first**2+last**2)/(2*np.where(total>0,total,1)))))

def coordinates(atoms,mass,lower,upper):
    return np.concatenate((2*(np.asarray(atoms)-lower)/(upper-lower)-1,
                           (2*np.asarray(mass)-1)[...,None]),-1)

def unpack(value,lower,upper):
    return lower+(value[...,:2]+1)*(upper-lower)/2,(value[...,2]+1)/2

def project(value):
    assert value.shape[-1]==3 and torch.isfinite(value).all()
    first=value[...,:2];mean=first.mean(-1,keepdim=True).expand_as(first)
    ordered=torch.where((first[...,0]>first[...,1])[...,None],mean,first).clamp(-1,1)
    return torch.cat((ordered,value[...,2:].clamp(-1,1)),-1)

def encode(value):
    assert value.shape[-1]==3 and torch.isfinite(value).all()
    assert ((value>=-1)&(value<=1)).all() and (value[...,0]<=value[...,1]).all()
    return torch.cat((value,torch.ones_like(value[...,:1])),-1)*(2/(1+value.square().sum(-1,keepdim=True))).sqrt()

def decode(received):
    assert received.shape[-1]==4 and torch.isfinite(received).all()
    return project(received[...,:3]/received[...,3:].clamp_min(2**-.5))
