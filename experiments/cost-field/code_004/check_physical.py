"""Full transmitted/fading/noise/baseband records are independently replayable."""
import json
from pathlib import Path
import torch
from codec import physical_channel,receive_project
def main():
 torch.manual_seed(17);tx=torch.randn(1,114,2,dtype=torch.float64)
 rows=[]
 for kind in ('identity','awgn','rayleigh'):
  received,info=physical_channel(tx,kind,10,torch.Generator().manual_seed(17))
  xr,xi=tx.unbind(-1);hr,hi=info['fading'].unbind(-1);nr,ni=info['noise'].unbind(-1)
  yr=hr*xr-hi*xi+nr;yi=hr*xi+hi*xr+ni
  assert torch.equal(torch.stack((yr,yi),-1),info['received_baseband'])
  denom=hr*hr+hi*hi
  actual=torch.stack(((hr*yr+hi*yi)/denom,(hr*yi-hi*yr)/denom),-1)
  assert torch.equal(received,actual)
  assert torch.isfinite(receive_project(received.reshape(1,19,6,2),6)).all()
  rows.append(dict(channel=kind,complex_uses=114,noise_values=info['noise'].numel(),fading_values=info['fading'].numel(),baseband_values=info['received_baseband'].numel()))
 r=dict(state='passed_complete_physical_array_replay_contract',rows=rows,limitation='CPU exact arithmetic contract only; full formal training remains unrun.')
 out=Path(__file__).resolve().parents[3]/'data/engineering/cost-field-physical-contract-004.json';assert not out.exists();out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
