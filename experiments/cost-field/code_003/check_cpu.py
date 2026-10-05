"""Coordinate and receiver contracts before actual native optimizer updates."""
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from codec import CostFieldCodec,GenericDenseCodec,receive_project,physical_channel

def main():
 torch.set_num_threads(2);torch.manual_seed(17)
 axis=torch.linspace(2,59.6,72,dtype=torch.float64)
 sweep=torch.arange(72,dtype=torch.float64)*.8+2.4
 # The unchanged native normalization and align_corners=True consume vertices,
 # verified using a linear physical-depth field at arbitrary metric queries.
 queries=torch.tensor([2,2.4,7.3,18,40,59.2,59.6],dtype=torch.float64)
 grid=torch.zeros(1,1,1,len(queries),3,dtype=torch.float64)
 grid[...,2]=((queries-2)/57.6*2-1)
 sampled=F.grid_sample(axis.reshape(1,1,72,1,1),grid,align_corners=True).ravel()
 assert torch.max(torch.abs(sampled-queries))<1e-12
 old=F.grid_sample(sweep.reshape(1,1,72,1,1),grid,align_corners=True).ravel()
 assert abs(float(old[0]-queries[0])-.4)<1e-12 and abs(float(old[-1]-queries[-1])+.4)<1e-12
 projection=[]
 for scale in (0.,1.,1e30):
  raw=(torch.randn(2,4,2)*scale).requires_grad_()
  y=receive_project(raw,4);y.sum().backward()
  assert torch.isfinite(y).all() and torch.isfinite(raw.grad).all()
  assert torch.allclose(y.square().sum((-2,-1)),torch.full((2,),4.),atol=2e-6,rtol=0)
  assert torch.allclose(receive_project(y.detach(),4),y.detach(),atol=1e-6,rtol=0)
  projection.append(dict(scale=scale,max_absolute=float(y.detach().abs().max())))
 # Physical deep-fade observation stays huge; only decoder-domain projection
 # uses the known source sphere. No channel/fade value is modified.
 h=torch.tensor(1e-8,dtype=torch.float64);tx=torch.ones(1,4,2,dtype=torch.float64)
 noisy=h*tx+torch.ones_like(tx);rx=noisy/h;copy=rx.clone()
 assert rx.abs().max()>1e7
 assert receive_project(rx,4).abs().max()<=2 and torch.equal(rx,copy)
 cost=torch.randn(1,32,72,8,12);app=torch.randn(1,32,8,12)
 prob=torch.randn(1,1,72,8,12).softmax(2)
 rows=[]
 for arm in ('G','P','S','B'):
  cls=GenericDenseCodec if arm=='B' else CostFieldCodec
  codec=cls(axis.float(),arm)
  packet,private=codec.encode(cost,app,prob);assert packet.shape==(1,114,2)
  assert abs(float(packet.detach().double().square().sum())-114)<114e-6
  for channel in ('identity','awgn','rayleigh'):
   codec.zero_grad(set_to_none=True)
   # Fresh encode graphs for each backward, no reused or in-place private state.
   packet,private=codec.encode(cost,app,prob)
   received,_=physical_channel(packet,channel,10,torch.Generator().manual_seed(19))
   rec,appearance=codec.decode(received,(8,12),(8,12));private.clear()
   other=cls(axis.float(),arm);other.load_state_dict(codec.state_dict())
   with torch.no_grad():
    for name,p in other.named_parameters():
     if name.startswith(('score.','cost_encoder.','depth_encoder.','appearance_encoder.')):p.normal_()
   repeated=other.decode(received.detach(),(8,12),(8,12))
   assert torch.equal(repeated[0],rec.detach()) and torch.equal(repeated[1],appearance.detach())
   (rec.square().mean()+appearance.square().mean()).backward()
   assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in codec.parameters())
   for name in ('cost_encoder','cost_decoder','appearance_encoder','appearance_decoder'):
    assert torch.count_nonzero(getattr(codec,name).weight.grad)>0
   if arm=='B':
    assert torch.count_nonzero(codec.depth_encoder.weight.grad)>0 and torch.count_nonzero(codec.depth_decoder.weight.grad)>0
   rows.append(dict(arm=arm,channel=channel,parameters=sum(p.numel() for p in codec.parameters()),complex_uses=114))
  assert sum(p.numel() for p in codec.parameters())==(1751 if arm=='B' else 1650)
 result=dict(state='passed_native_query_coordinate_sphere_projection_equal_budget_and_generic_control_contracts',rows=rows,projection=projection,coordinate_max_error=float((sampled-queries).abs().max()),old_sweep_endpoint_bias=[float(old[0]-queries[0]),float(old[-1]-queries[-1])],limitation='Synthetic CPU contracts; no native GPU optimization or AP benefit.')
 root=Path(__file__).resolve().parents[3];out=root/'data/engineering/cost-field-CPU-contract-003.json'
 assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
