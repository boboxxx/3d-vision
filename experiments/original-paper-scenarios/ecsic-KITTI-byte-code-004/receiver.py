"""Fresh P6EK-only registered-model receiver; all causal arrays and public crop."""
import argparse,hashlib,io,json,os,time
from pathlib import Path
import numpy as np
import torch
import codec as c
import neural as n

def describe(v):
 a=np.ascontiguousarray(v);return dict(dtype=str(a.dtype),shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest())
def main():
 p=argparse.ArgumentParser()
 for key in ['registry','container','directory']:p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--registry-sha',required=True);p.add_argument('--candidate',type=int,choices=range(6),required=True);a=p.parse_args();assert not a.directory.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(2)
 def barrier(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)):
   name=os.fsdecode(args[0]);mode,flags=args[1:3];reading=(isinstance(mode,str) and 'r' in mode) or (mode is None and flags&os.O_ACCMODE==os.O_RDONLY)
   if reading:assert '/data/kitti/' not in name and not any(v in name for v in ['/label_2/','/calib/','/velodyne/']) and not name.endswith('.npz'),'Receiver attempted source/nonpublic inputs'
 import sys;sys.addaudithook(barrier)
 assert n.sha(a.registry)==a.registry_sha;registry=json.loads(a.registry.read_text());entry=registry['entries'][a.candidate];assert a.container.stat().st_size<=c.LIMIT;blob=a.container.read_bytes();parsed=c.unpack(blob,entry['model_sha256'])
 model,utils,entry=n.load(a.registry,a.registry_sha,a.candidate);before=n.states(model.state_dict());counts,handles=n.hooks(model,receiver=True);started=time.time()
 arrays,bins=n.decode(model,utils,parsed);assert counts==dict(E=0,HE=0,HD=1,D=1) and n.states(model.state_dict())==before and all(v.grad is None for v in model.parameters());h,w=parsed['original_hw'];cropped={side:np.ascontiguousarray(arrays['pred_'+side][:,:,:h,:w]) for side in ['left','right']}
 a.directory.mkdir(parents=True)
 def save_arrays(path,values):
  buffer=io.BytesIO();np.savez(buffer,**values);b=buffer.getvalue();path.write_bytes(b);return hashlib.sha256(b).hexdigest()
 output=save_arrays(a.directory/'causal-arrays.npz',arrays);crop=save_arrays(a.directory/'received-RGB.npz',cropped)
 result=dict(state='finished_model_bound_byte_only_neural_reception_pending_independent_array_audit',candidate=a.candidate,registry_sha256=a.registry_sha,model_sha256=entry['model_sha256'],container_sha256=hashlib.sha256(blob).hexdigest(),container_bytes=len(blob),original_hw=parsed['original_hw'],padded_hw=parsed['padded_hw'],calls=counts,states_before=before,states_after=n.states(model.state_dict()),all_gradients_absent=True,source_read_barrier=True,arrays={k:describe(v) for k,v in arrays.items()},scale_bins={k:describe(v) for k,v in bins.items()},native_crops={k:describe(v) for k,v in cropped.items()},causal_arrays_file_sha256=output,received_RGB_file_sha256=crop,torch_version=torch.__version__,numpy_version=np.__version__,elapsed_seconds=time.time()-started,AP_executed=False)
 (a.directory/'receiver.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(state=result['state'],container_bytes=len(blob),arrays=len(arrays))))
if __name__=='__main__':main()
