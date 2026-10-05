"""Conditional depth/feature packets with an explicit received-only decoder."""
import math
import torch
from torch import nn
import torch.nn.functional as F

def interval_masses(p, k=3):
 """p[B,1,D,H,W] -> overlap of CDF atoms with equal mass intervals."""
 cdf=p.cumsum(2);previous=cdf-p
 low=torch.arange(k,device=p.device,dtype=p.dtype)[None,:,None,None,None]/k
 high=low+1/k
 return (torch.minimum(cdf,high)-torch.maximum(previous,low)).clamp_min(0)

def barycenters(p, features, axis, k=3):
 mass=interval_masses(p,k)
 # Normalize floating-point CDF overlap residuals; exact mathematical mass is1/K.
 weights=mass/mass.sum(2,keepdim=True).clamp_min(torch.finfo(p.dtype).tiny)
 mu=torch.einsum('bkdhw,d->bkhw',weights,axis)
 values=torch.einsum('bkdhw,bcdhw->bkchw',weights,features)
 return mu,values

def normalize_symbols(x, energy):
 norm=x.square().sum((-2,-1),keepdim=True).clamp_min(torch.finfo(x.dtype).eps**2).sqrt()
 small=norm<=torch.finfo(x.dtype).eps
 fallback=torch.zeros_like(x);fallback[...,0,0]=math.sqrt(energy)
 return torch.where(small,fallback,x/norm.clamp_min(torch.finfo(x.dtype).eps)*math.sqrt(energy))

def physical_channel(tx,kind,snr_db,generator=None):
 if kind=='identity':return tx,dict(channel=kind,complex_uses=tx.shape[1],energy=tx.square().sum((1,2)))
 if kind not in ('awgn','rayleigh'):raise ValueError(kind)
 sigma=math.sqrt(10**(-float(snr_db)/10)/2)
 noise=torch.randn(tx.shape,device=tx.device,dtype=tx.dtype,generator=generator)*sigma
 if kind=='awgn':rx=tx+noise
 else:
  h=torch.randn(tx.shape,device=tx.device,dtype=tx.dtype,generator=generator)/math.sqrt(2)
  hr,hi=h.unbind(-1);xr,xi=tx.unbind(-1)
  yr=hr*xr-hi*xi+noise[...,0];yi=hr*xi+hi*xr+noise[...,1]
  norm=hr.square()+hi.square();assert (norm>0).all()
  rx=torch.stack(((hr*yr+hi*yi)/norm,(hr*yi-hi*yr)/norm),-1)
 return rx,dict(channel=kind,complex_uses=tx.shape[1],energy=tx.square().sum((1,2)),
  receiver_CSI='perfect_iid_no_fade_clip' if kind=='rayleigh' else 'not_needed')

class CostFieldCodec(nn.Module):
 def __init__(self,depth_axis,arm,k=3,stride=4):
  super().__init__();assert arm in ('G','P','S')
  self.arm,self.k,self.stride=arm,k,stride
  self.register_buffer('depth_axis',depth_axis.detach().clone())
  self.score=nn.Sequential(nn.Conv3d(32,16,1),nn.ReLU(),nn.Conv3d(16,1,1))
  nn.init.zeros_(self.score[-1].weight);nn.init.zeros_(self.score[-1].bias)
  self.cost_encoder=nn.Conv3d(32,8,1);self.cost_decoder=nn.Conv2d(8,32,1)
  self.appearance_encoder=nn.Conv2d(32,8,1);self.appearance_decoder=nn.Conv2d(8,32,1)
  self.log_bandwidth=nn.Parameter(torch.log((depth_axis[-1]-depth_axis[0])/k))
 def encode(self,cost,appearance,native_probability):
  assert cost.shape[1]==appearance.shape[1]==32 and cost.shape[2]==len(self.depth_axis)
  cost=F.avg_pool3d(cost,(1,self.stride,self.stride))
  appearance=F.avg_pool2d(appearance,self.stride)
  prior=F.avg_pool3d(native_probability,(1,self.stride,self.stride))
  assert prior.shape==(cost.shape[0],1,*cost.shape[2:]) and appearance.shape[-2:]==cost.shape[-2:]
  prior=prior/prior.sum(2,keepdim=True)
  if self.arm=='S':prior=prior.roll((prior.shape[-2]//2,prior.shape[-1]//2),(-2,-1))
  logits=self.score(cost)
  if self.arm!='G':logits=logits+prior.clamp_min(torch.finfo(cost.dtype).tiny).log()
  p=logits.softmax(2)
  mu,features=barycenters(p,self.cost_encoder(cost),self.depth_axis,self.k)
  lo,hi=self.depth_axis[0],self.depth_axis[-1]
  angle=((mu-lo)/(hi-lo)*2-1)*(math.pi/2)
  positions=torch.stack((angle.cos(),angle.sin()),-1).permute(0,2,3,1,4)
  b,k,c,h,w=features.shape
  values=features.permute(0,3,4,1,2).reshape(b,h,w,k,4,2)
  values=normalize_symbols(values,4)
  app=self.appearance_encoder(appearance).permute(0,2,3,1).reshape(b,h,w,4,2)
  app=normalize_symbols(app,4)
  packet=torch.cat([positions,values.flatten(3,4),app],3).reshape(b,h*w*(self.k*5+4),2)
  return packet,dict(pooled_hw=(h,w),native_cost_shape=tuple(cost.shape),mu=mu,p=p)
 def decode(self,received,cost_hw,appearance_hw):
  """No clean features, source scales, probabilities, means or masks accepted."""
  h,w=cost_hw[0]//self.stride,cost_hw[1]//self.stride;b=received.shape[0]
  assert received.shape==(b,h*w*(self.k*5+4),2)
  payload=received.reshape(b,h,w,self.k*5+4,2)
  angle=torch.atan2(payload[...,:self.k,1],payload[...,:self.k,0]).clamp(-math.pi/2,math.pi/2)
  lo,hi=self.depth_axis[0],self.depth_axis[-1]
  mu=(angle/(math.pi/2)+1)/2*(hi-lo)+lo
  value=payload[...,self.k:self.k*5,:].reshape(b,h,w,self.k,8)
  value=value.permute(0,3,4,1,2).reshape(b*self.k,8,h,w)
  decoded=self.cost_decoder(value).reshape(b,self.k,32,h,w)
  delta=self.depth_axis[None,None,:,None,None]-mu.permute(0,3,1,2)[:,:,None]
  bandwidth=self.log_bandwidth.exp().clamp_min(torch.finfo(received.dtype).eps)
  weight=(-.5*(delta/bandwidth).square()).exp()/self.k
  cost=torch.einsum('bkdhw,bkchw->bcdhw',weight,decoded)
  cost=F.interpolate(cost,size=(len(self.depth_axis),*cost_hw),mode='trilinear',align_corners=True)
  app=payload[...,self.k*5:,:].reshape(b,h,w,8).permute(0,3,1,2)
  app=self.appearance_decoder(app)
  app=F.interpolate(app,size=appearance_hw,mode='bilinear',align_corners=True)
  return cost,app
