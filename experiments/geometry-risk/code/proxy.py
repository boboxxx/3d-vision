"""Sensor feature arithmetic only; no detector, GT, fitting or calibration claim."""
import copy
import math
import numpy as np


def feature(value):
    value=np.asarray(value)
    if value.dtype not in (np.dtype('float32'),np.dtype('float64')) or value.ndim!=3 or value.shape[0]!=32 or min(value.shape[1:])<8 or any(v%4 for v in value.shape[1:]) or not np.isfinite(value).all():
        raise ValueError('finite32xHxW float stereo feature, H/W divisible by4')
    c,h,w=value.shape
    return value.astype(np.float64).reshape(c,h//4,4,w//4,4).mean((2,4))


def normalize(value,energy_threshold):
    norms=np.linalg.norm(value,axis=0)
    valid=norms>energy_threshold
    out=np.divide(value,norms[None],out=np.zeros_like(value),where=valid[None])
    texture=np.zeros(norms.shape,dtype=np.float64)
    delta=np.sum((out[:,:,1:]-out[:,:,:-1])**2,axis=0)
    texture[:,1:]+=delta;texture[:,:-1]+=delta
    delta=np.sum((out[:,1:,:]-out[:,:-1,:])**2,axis=0)
    texture[1:,:]+=delta;texture[:-1,:]+=delta
    return out,valid & (texture>1e-12)


def summarize(scores,support,fb):
    valid=support.any(0)
    masked=np.where(support,scores/.1,-np.inf)
    maximum=np.max(masked,axis=0,initial=-np.inf)
    maximum=np.where(valid,maximum,0)
    values=np.exp(masked-maximum[None]);denom=values.sum(0)
    probability=np.divide(values,denom[None],out=np.zeros_like(values),where=denom[None]>0)
    disparities=4.0*np.arange(1,49)[:,None,None]
    depths=fb/disparities
    mean=(probability*depths).sum(0)
    variance=(probability*(depths-mean[None])**2).sum(0)
    dmean=(probability*disparities).sum(0)
    dvar=(probability*(disparities-dmean[None])**2).sum(0)
    logp=np.log(probability,out=np.zeros_like(probability),where=probability>0)
    result=dict(probability=probability,valid=valid,support_count=support.sum(0),
                entropy=-(probability*logp).sum(0),depth_mean=mean,depth_variance=variance,
                disparity_mean=dmean,disparity_variance=dvar)
    for key in ('entropy','depth_mean','depth_variance','disparity_mean','disparity_variance'):
        result[key]=np.where(valid,result[key],np.nan)
    return result


def epipolar_proxy(left,right,*,focal_baseline,valid_image_shape=None):
    if type(focal_baseline) not in (int,float) or not math.isfinite(focal_baseline) or focal_baseline<=0:
        raise ValueError('finite positive public scalar focal-baseline, image pixel*m')
    thresholds=(np.finfo(np.asarray(left).dtype).eps if np.asarray(left).dtype.kind=="f" else 0, np.finfo(np.asarray(right).dtype).eps if np.asarray(right).dtype.kind=="f" else 0)
    original_shape=np.asarray(left).shape[-2:]
    left,right=feature(left),feature(right)
    if left.shape!=right.shape:raise ValueError('matched stereo geometry')
    left,vl=normalize(left,thresholds[0]);right,vr=normalize(right,thresholds[1])
    _,h,w=left.shape
    if valid_image_shape is not None:
        if (type(valid_image_shape) is not tuple or len(valid_image_shape)!=2
                or any(type(x) is not int or x<=0 or x>maximum for x,maximum in zip(valid_image_shape,original_shape))):
            raise ValueError('public cropped height/width tuple required, no arbitrary spatial mask')
        extent=(np.arange(h)[:,None]<valid_image_shape[0]//4)&(np.arange(w)[None]<valid_image_shape[1]//4)
        vl&=extent;vr&=extent
    sl=np.zeros((48,h,w));sr=np.zeros_like(sl);ml=np.zeros_like(sl,dtype=bool);mr=np.zeros_like(ml)
    for d in range(1,min(48,w-1)+1):
        correlation=(left[:,:,d:]*right[:,:,:-d]).sum(0)
        valid=vl[:,d:] & vr[:,:-d]
        sl[d-1,:,d:]=correlation;ml[d-1,:,d:]=valid
        sr[d-1,:,:-d]=correlation;mr[d-1,:,:-d]=valid
    return {'left':summarize(sl,ml,focal_baseline),'right':summarize(sr,mr,focal_baseline),
            'disparity_bins_input_pixels':4.0*np.arange(1,49),'focal_baseline':float(focal_baseline),'valid_image_shape':valid_image_shape,
            'scope':'coarse_cosine_probe_uncalibrated_intrinsic_ambiguity_not_channel_damage'}


def groups(layouts):
    ids=[]
    for b,c,h,w in layouts:
        if b!=1 or c%2 or min(c,h,w)<=0:raise ValueError('batch1 even positive complex-code layout')
        row=np.minimum(((np.arange(h)+.5)*4/h).astype(int),3)
        col=np.minimum(((np.arange(w)+.5)*8/w).astype(int),7)
        # Existing pack order is spatial cell, then consecutive channel I/Q pair.
        ids.append(np.repeat((row[:,None]*8+col[None,:]).reshape(-1),c//2))
    return np.concatenate(ids).astype(np.int64)


def group_noise(group_ids,group,*,rng):
    if not isinstance(rng.bit_generator,np.random.PCG64) or type(group) is not int or not 0<=group<32:
        raise ValueError('group0..31 and dedicated PCG64')
    group_ids=np.asarray(group_ids)
    if group_ids.ndim!=1 or not np.isin(group_ids,np.arange(32)).all():raise ValueError('valid public group mapping')
    before=copy.deepcopy(rng.bit_generator.state)
    noise=rng.standard_normal((len(group_ids),2))*math.sqrt(.1/2)
    noise[group_ids!=group]=0
    return noise.astype(np.float32),dict(before=before,after=copy.deepcopy(rng.bit_generator.state),draw_shape=[len(group_ids),2],selected_symbols=int(np.count_nonzero(group_ids==group)),attempted_complex_uses=len(group_ids),local_N0=.1)
