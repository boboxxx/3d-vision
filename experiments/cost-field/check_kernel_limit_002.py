"""Reproduce cancellation and verify the locked conditional-measure correction."""
import importlib.util
import json
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def load(p,name):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m.CostFieldCodec
def main():
 torch.set_num_threads(2);torch.manual_seed(37)
 axis=torch.linspace(2.4,59.2,9,dtype=torch.float64)
 normalized=load(HERE/'codec.py','normalized_codec')(axis,'P').double().eval()
 measure=load(HERE/'code_002/codec.py','measure_codec')(axis,'P').double().eval()
 measure.load_state_dict(normalized.state_dict())
 app=torch.randn(1,32,8,12,dtype=torch.float64);feature=torch.randn(1,32,1,1,1,dtype=torch.float64)
 outputs={name:[] for name in ('normalized','measure')};packet_equality=[];peaks=[]
 for index in (0,8):
  cost=torch.zeros(1,32,9,8,12,dtype=torch.float64);cost[:,:,index:index+1]=feature
  p=torch.zeros(1,1,9,8,12,dtype=torch.float64);p[:,:,index]=1
  with torch.no_grad():
   tx1,_=normalized.encode(cost,app,p);tx2,_=measure.encode(cost,app,p)
   assert torch.equal(tx1,tx2);packet_equality.append(True)
   for name,model,tx in [('normalized',normalized,tx1),('measure',measure,tx2)]:
    decoded,_=model.decode(tx,(8,12),(8,12));outputs[name].append(decoded)
   profile=outputs['measure'][-1].square().sum(1).mean((0,2,3))
   peak=int(profile.argmax());assert peak==index;peaks.append(peak)
 old=float((outputs['normalized'][0]-outputs['normalized'][1]).abs().max())
 corrected=float((outputs['measure'][0]-outputs['measure'][1]).abs().max())
 assert old<1e-12 and corrected>torch.finfo(torch.float64).eps
 result=dict(state='passed_conditional_measure_preserves_single_mode_depth_with_identical_packets_and_parameters',
  normalized_RBF_max_difference=old,corrected_measure_max_difference=corrected,
  source_depth_difference_m=float(axis[-1]-axis[0]),corrected_peak_indices=peaks,
  all_packets_identical_between_decoders=packet_equality,
  limitation='Structural synthetic counterexample and correction only; no trained detection or geometry AP claim')
 target=ROOT/'data/engineering/cost-field-kernel-limit-002.json';assert not target.exists()
 target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
