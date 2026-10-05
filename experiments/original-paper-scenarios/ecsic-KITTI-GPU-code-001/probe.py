"""Two full-native official ECSIC source-RD Adam steps; no detector/AP."""
import hashlib, importlib, json, os, subprocess, sys, time, traceback, types
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'data/engineering/ecsic-KITTI-GPU-001'
LOCK=HERE/'inputs.json'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def array(v):return v.detach().cpu().contiguous().numpy()
def descriptor(v):
 a=np.ascontiguousarray(v)
 return dict(dtype=str(a.dtype),shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest())
def states(s):return {k:descriptor(array(v)) for k,v in s.items()}
def freeze():
 env=os.environ.copy();env.pop('PYTHONPATH',None)
 return subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True,env=env)
def load_model():
 sys.path.insert(0,str(ROOT/'data/engineering/ecsic-import-overlay-001'))
 def disabled(*a,**k):raise RuntimeError('Remote logger/experiment manager forbidden')
 logger=types.ModuleType('wandb');logger.__file__='disabled-import-only-wandb'
 for k in ('init','log','watch','finish'):setattr(logger,k,disabled)
 logger.util=types.SimpleNamespace(generate_id=disabled);sys.modules['wandb']=logger
 package=types.ModuleType('ecsic');package.__path__=[str(ROOT/'third_party/ECSIC-official-001/ecsic')];sys.modules['ecsic']=package
 models=importlib.import_module('ecsic.models');utils=importlib.import_module('ecsic.utils');metrics=importlib.import_module('ecsic.metrics')
 cfg=json.loads((ROOT/'data/provenance/ecsic-cs001-official-config-001.json').read_text())['model']
 model=getattr(models,cfg['name'])(**cfg['kwargs'])
 path=ROOT/'assets/ecsic-cs001-001/model.pt';assert sha(path)=='e66b55e2f888f5d113ea6c6414640c7486280d6e18838ba77c650720a5636052'
 checkpoint=torch.load(path,map_location='cpu',weights_only=True)
 assert set(checkpoint)==set(model.state_dict()) and len(checkpoint)==225
 model.load_state_dict(checkpoint,strict=True);assert all(torch.equal(checkpoint[k],v) for k,v in model.state_dict().items())
 return model,utils,metrics

def main():
 assert not OUT.exists();OUT.mkdir(parents=True)
 report=dict(state='starting',scope='two_step_full_native_ECSIC_GPU_RD_engineering_only',job_id=os.environ.get('SLURM_JOB_ID'),steps=[],started_unix=time.time(),AP_measured=False,formal_training_closed=False)
 def save():(OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 save()
 try:
  lock=json.loads(LOCK.read_text());report['input_sha256']=sha(LOCK)
  identities={p:sha(ROOT/p) for p in lock['files']};assert identities==lock['files'];report['sources_before']=identities
  assert freeze()==lock['base_freeze'];report['base_freeze_before']=lock['base_freeze']
  torch.set_num_threads(2);torch.manual_seed(17);np.random.seed(17)
  assert torch.__version__=='2.7.1+cu128' and np.__version__=='1.26.4'
  assert torch.cuda.device_count()==1 and torch.cuda.get_device_capability()==(12,0)
  gpu=torch.cuda.get_device_name();assert 'RTX PRO 6000' in gpu
  report.update(GPU=gpu,capability=[12,0],torch=torch.__version__,numpy=np.__version__,initial_free_bytes=torch.cuda.mem_get_info()[0]);save()
  model,utils,metrics=load_model();model=model.cuda().train();parameters=dict(model.named_parameters())
  import einops,torchvision
  assert einops.__version__=='0.8.2' and torchvision.__version__=='0.22.1+cu128'
  assert 'artemis-framework-overlay-005/site-packages' in torchvision.__file__
  report.update(parameter_names=list(parameters),parameter_count=sum(p.numel() for p in parameters.values()),state_count=len(model.state_dict()),runtime=dict(einops=einops.__version__,torchvision=torchvision.__version__,einops_path=einops.__file__,torchvision_path=torchvision.__file__))
  optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
  initial=OUT/'initial.pt';torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict()),initial)
  report['initial_checkpoint_sha256']=sha(initial);report['initial_states']=states(model.state_dict());save()
  manifest=json.loads((ROOT/'data/provenance/cost-field-full-dataset-manifest-004.json').read_text())
  for step,frame in enumerate(('000000','000003')):
   assert frame in manifest['splits']['train']['ids'] and frame not in manifest['splits']['val']['ids']
   images=[];source={}
   for folder in ('image_2','image_3'):
    rel=f'data/kitti/training/{folder}/{frame}.png';path=ROOT/rel;expected=manifest['files'][rel]
    assert path.stat().st_size==expected['bytes'] and sha(path)==expected['sha256'];source[rel]=expected
    rgb=np.asarray(Image.open(path).convert('RGB')).copy();images.append(torch.from_numpy(rgb).permute(2,0,1)[None].cuda().float()/255)
   assert images[0].shape==images[1].shape;h,w=images[0].shape[-2:];ph,pw=(-h)%32,(-w)%32
   left,right=[F.pad(v,(0,pw,0,ph),mode='replicate') for v in images]
   pos=utils.get_positional_fourier_encoding(h+ph,w+pw).unsqueeze(0).cuda()
   before=states(model.state_dict());optimizer.zero_grad(set_to_none=True)
   start=time.time();output=model(left,right,pos)
   mse=(metrics.calc_mse(left,output.pred.left)+metrics.calc_mse(right,output.pred.right))/2
   ratevalues=[output.rate.left.y,output.rate.left.z,output.rate.right.y,output.rate.right.z]
   bpp=sum(metrics.calc_bpp(v,left) for v in ratevalues)/2
   loss=(bpp+.01*mse)/1.01;assert torch.isfinite(loss);loss.backward()
   arrays=dict(input_left=array(left),input_right=array(right),pred_left=array(output.pred.left),pred_right=array(output.pred.right),losses=np.asarray([float(mse.detach()),float(bpp.detach()),float(loss.detach())],np.float32))
   for name,v in zip(('left_y','left_z','right_y','right_z'),ratevalues):arrays['rate_'+name]=array(v)
   for view in ('left','right'):
    for key,value in output.latents[view].items():arrays[f'latent_{view}_{key}']=array(value)
   for name,p in parameters.items():
    assert p.grad is not None and torch.isfinite(p.grad).all(),name
    arrays['gradient_'+name]=array(p.grad).copy()
   norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),2.5))
   for name,p in parameters.items():arrays['clipped_gradient_'+name]=array(p.grad).copy()
   optimizer.step();torch.cuda.synchronize()
   assert all(torch.isfinite(v).all() for v in model.state_dict().values())
   assert all(np.isfinite(v).all() for v in arrays.values())
   path=OUT/f'step{step}.npz';np.savez_compressed(path,**arrays)
   checkpoint=OUT/f'step{step}-state.pt';torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict()),checkpoint)
   report['steps'].append(dict(step=step,frame=frame,lambda_RD=.01,original_hw=[h,w],padded_hw=[h+ph,w+pw],source=source,arrays_path=str(path.relative_to(ROOT)),arrays_sha256=sha(path),arrays={k:descriptor(v) for k,v in arrays.items()},states_before=before,states_after=states(model.state_dict()),checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=sha(checkpoint),gradient_norm=norm,wall_seconds=time.time()-start))
   del arrays,output,images,left,right,mse,bpp,loss
   save();print(json.dumps(dict(step=step,frame=frame,gradient_norm=norm)),flush=True)
  report.update(sources_after={p:sha(ROOT/p) for p in identities},base_freeze_after=freeze(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
  assert report['sources_after']==identities and report['base_freeze_after']==lock['base_freeze']
  assert report['peak_reserved_bytes']+2*1024**3<=report['initial_free_bytes']
  report['state']='finished_two_native_RD_updates_pending_actual_terminal_and_complete_audit'
 except BaseException:
  report.update(state='failed',traceback=traceback.format_exc());raise
 finally:report['finished_unix']=time.time();save()

if __name__=='__main__':main()
