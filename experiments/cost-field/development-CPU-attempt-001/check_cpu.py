"""Independent math/resource/receiver/gradient tests for actual operator contracts."""
import json
from pathlib import Path
import numpy as np
import torch
from codec import CostFieldCodec,barycenters,physical_channel,normalize_symbols

def main():
 torch.set_num_threads(2);torch.manual_seed(17)
 axis=torch.linspace(2.4,59.2,9,dtype=torch.float64)
 p=torch.tensor([.06,.19,.04,.11,.08,.17,.04,.23,.08],dtype=torch.float64)[None,None,:,None,None]
 f=torch.arange(18,dtype=torch.float64).reshape(1,2,9,1,1)
 mu,values=barycenters(p,f,axis)
 cdf=np.cumsum(p.numpy().ravel());previous=np.r_[0,cdf[:-1]]
 # Independent scalar overlap/integration; no calls to the tested torch operator.
 means=[];features=[]
 for k in range(3):
  weights=np.array([max(0,min(cdf[j],(k+1)/3)-max(previous[j],k/3)) for j in range(9)])*3
  means.append(sum(weights[j]*axis[j].item() for j in range(9)))
  features.append([sum(weights[j]*f[0,c,j,0,0].item() for j in range(9)) for c in range(2)])
 assert np.max(np.abs(mu.numpy().ravel()-means))<1e-12
 assert np.max(np.abs(values.numpy().reshape(3,2)-features))<1e-12
 # Cross a former hard-quantile jump: barycenters remain continuous.
 a=torch.tensor([1/3-1e-8,2/3+1e-8],dtype=torch.float64)[None,None,:,None,None]
 b=torch.tensor([1/3+1e-8,2/3-1e-8],dtype=torch.float64)[None,None,:,None,None]
 m1,_=barycenters(a,torch.ones(1,2,2,1,1,dtype=torch.float64),torch.tensor([4.,52.],dtype=torch.float64))
 m2,_=barycenters(b,torch.ones(1,2,2,1,1,dtype=torch.float64),torch.tensor([4.,52.],dtype=torch.float64))
 assert (m1-m2).abs().max()<3e-6
 # Native-shape spatial grouping, equal parameters and exact complex budget.
 cost=torch.randn(1,32,9,8,12,dtype=torch.float64);app=torch.randn(1,32,8,12,dtype=torch.float64)
 prob=torch.randn(1,1,9,8,12,dtype=torch.float64).softmax(2)
 codecs={arm:CostFieldCodec(axis,arm).double() for arm in ('G','P','S')}
 base=codecs['G'].state_dict()
 for model in codecs.values():model.load_state_dict(base)
 assert len({sum(p.numel() for p in m.parameters()) for m in codecs.values()})==1
 rows=[]
 for arm,model in codecs.items():
  tx,private=model.encode(cost,app,prob);assert tx.shape==(1,114,2)
  assert abs(float(tx.square().sum())-114)<1e-10
  rx,_=physical_channel(tx,'awgn',10,torch.Generator().manual_seed(731))
  rec,a=model.decode(rx,(8,12),(8,12));assert rec.shape==cost.shape and a.shape==app.shape
  # Receiver output is invariant to any later sender-private mutation.
  first=(rec.detach().clone(),a.detach().clone());private['mu'].fill_(0);private['p'].fill_(0)
  c2,a2=model.decode(rx,(8,12),(8,12));assert torch.equal(c2,first[0]) and torch.equal(a2,first[1])
  (rec.square().mean()+a.square().mean()).backward()
  gradients=[p.grad for p in model.parameters()];assert all(g is not None and torch.isfinite(g).all() for g in gradients)
  assert all((getattr(model,name).weight.grad!=0).any() for name in ('cost_encoder','cost_decoder','appearance_encoder','appearance_decoder'))
  faded,physical=physical_channel(tx.detach(),'rayleigh',10,torch.Generator().manual_seed(91));assert torch.isfinite(faded).all()
  rows.append(dict(arm=arm,complex_uses=114,energy=float(tx.detach().square().sum()),parameter_count=sum(p.numel() for p in model.parameters())))
 zero=normalize_symbols(torch.zeros(2,4,2),4);assert torch.equal(zero.square().sum((1,2)),torch.full((2,),4.))
 result=dict(state='passed_independent_barycenters_continuity_resource_receiver_and_gradient_contracts',rows=rows,
  limitation='CPU synthetic contract checks only; no native detector execution, training or scientific benefit')
 out=Path(__file__).resolve().parents[2]/'data/engineering/cost-field-CPU-contract-001.json';assert not out.exists()
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
