"""Six rate operating models with a single seed17; native RGB source-RD."""
import argparse,hashlib,io,json,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'ecsic-KITTI-GPU-code-001'))
from probe import load_model,sha,states,freeze
LAMBDAS=(.001,.003,.01,.03,.1,.3)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--candidate',type=int,choices=range(6),required=True);args=parser.parse_args()
 value=LAMBDAS[args.candidate];tag=format(value,'.3g')
 out=ROOT/'data/runs'/f'ecsic-KITTI-lambda{tag}-seed17-002';assert not out.exists();out.mkdir(parents=True)
 r=dict(state='starting',job_id=os.environ.get('SLURM_JOB_ID'),array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),candidate=args.candidate,lambda_RD=value,seed=17,update_count=0,completed_epochs=[],started_unix=time.time(),no_detector_or_GT=True,formal_raw_gradient_replay=False)
 def save():
  temp=out/'report.tmp';temp.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');os.replace(temp,out/'report.json')
 save()
 try:
  lockpath=HERE/'inputs.json';lock=json.loads(lockpath.read_text());assert lock['lambdas']==list(LAMBDAS);r['input_sha256']=sha(lockpath)
  before={p:sha(ROOT/p) for p in lock['files']};assert before==lock['files'];assert freeze()==lock['base_freeze']
  r.update(sources_before=before,base_freeze_before=lock['base_freeze'])
  proofs=[]
  for path,digest in lock['engineering_proofs'].items():
   assert sha(ROOT/path)==digest;proof=json.loads((ROOT/path).read_text())
   assert proof['state']=='passed_complete_two_native_RGB_RD_Adam_steps' and proof['checked_steps']==2 and proof['complete_saved_values']==144085094 and proof['gradient_values']==126562968
   proofs.append(proof)
  assert proofs[0]['metadata']==proofs[1]['metadata']
  terminalpath=ROOT/lock['engineering_terminal_path'];assert sha(terminalpath)==lock['engineering_terminal_sha256'];terminal=json.loads(terminalpath.read_text());assert terminal['state']=='closed_actual_GPU_native_and_local_complete_two_update_ECSIC_gate'
  manifestpath=ROOT/'data/provenance/cost-field-full-dataset-manifest-004.json';manifest=json.loads(manifestpath.read_text())
  ids=manifest['splits']['train']['ids'];assert len(ids)==len(set(ids))==3712 and not set(ids)&set(manifest['splits']['val']['ids'])
  allowed={str((ROOT/f'data/kitti/training/{folder}/{frame}.png').resolve()) for frame in ids for folder in ('image_2','image_3')}
  def barrier(event,args):
   if event=='open' and isinstance(args[0],(str,bytes)):
    name=os.fsdecode(args[0]);mode,flags=args[1:3]
    reading=(isinstance(mode,str) and 'r' in mode) or (mode is None and flags&os.O_ACCMODE==os.O_RDONLY)
    if reading and ('/data/kitti/' in name or any(s in name for s in ('/label_2/','/velodyne/','/calib/'))):assert str(Path(name).resolve()) in allowed,'Forbidden non-RGB/non-training sensor read'
  sys.addaudithook(barrier)
  torch.set_num_threads(2);torch.manual_seed(17);np.random.seed(17)
  assert torch.__version__=='2.7.1+cu128' and np.__version__=='1.26.4' and torch.cuda.device_count()==1 and torch.cuda.get_device_capability()==(12,0)
  gpu=torch.cuda.get_device_name();assert 'RTX PRO 6000' in gpu
  free=torch.cuda.mem_get_info()[0];model,utils,metrics=load_model();model=model.cuda().train();parameters=list(model.parameters())
  assert len(model.state_dict())==225 and sum(p.numel() for p in parameters)==31640742
  optimizer=torch.optim.Adam(parameters,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
  r.update(GPU=gpu,capability=[12,0],initial_free_bytes=free,initial_states=states(model.state_dict()),parameter_count=31640742,parameter_tensors=len(parameters),state_tensors=225,initialization_sha256=sha(ROOT/'assets/ecsic-cs001-001/model.pt'),manifest_sha256=sha(manifestpath),input_barrier=True,initial_CPU_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),initial_CUDA_rng_sha256=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest())
  record=out/'updates.jsonl';ledgerhash=hashlib.sha256();pos_cache={};save()
  with record.open('xb',buffering=0) as stream:
   for epoch in range(10):
    order=np.random.default_rng(17+1000*epoch).permutation(ids).tolist()
    for index,frame in enumerate(order):
     pair=[];source={}
     for folder in ('image_2','image_3'):
      rel=f'data/kitti/training/{folder}/{frame}.png';b=(ROOT/rel).read_bytes();item=manifest['files'][rel]
      assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256'];source[rel]=item['sha256']
      rgb=np.asarray(Image.open(io.BytesIO(b)).convert('RGB')).copy();pair.append(torch.from_numpy(rgb).permute(2,0,1)[None].cuda().float()/255.)
     assert pair[0].shape==pair[1].shape;h,w=pair[0].shape[-2:];ph,pw=(-h)%32,(-w)%32
     left,right=[F.pad(v,(0,pw,0,ph),mode='replicate') for v in pair]
     shape=(h+ph,w+pw)
     if shape not in pos_cache:pos_cache[shape]=utils.get_positional_fourier_encoding(*shape).unsqueeze(0).cuda()
     optimizer.zero_grad(set_to_none=True);output=model(left,right,pos_cache[shape])
     mse=(metrics.calc_mse(left,output.pred.left)+metrics.calc_mse(right,output.pred.right))/2
     bpp=sum(metrics.calc_bpp(v,left) for v in (output.rate.left.y,output.rate.left.z,output.rate.right.y,output.rate.right.z))/2
     loss=(bpp+value*mse)/(1+value);assert torch.isfinite(loss);loss.backward()
     assert all(p.grad is not None for p in parameters)
     assert torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all()
     norm=float(torch.nn.utils.clip_grad_norm_(parameters,2.5));assert np.isfinite(norm)
     optimizer.step()
     assert torch.stack([torch.isfinite(p).all() for p in parameters]).all()
     assert torch.stack([torch.isfinite(s[k]).all() for s in optimizer.state.values() for k in ('exp_avg','exp_avg_sq')]).all()
     step=epoch*3712+index;steps=[float(s['step']) for s in optimizer.state.values()];assert len(steps)==len(parameters) and min(steps)==max(steps)==step+1
     row=dict(step=step,epoch=epoch+1,index=index,frame_id=frame,lambda_RD=value,original_hw=[h,w],padded_hw=list(shape),source_png_sha256=source,mse_255scale=float(mse.detach()),estimated_bpp_per_view=float(bpp.detach()),loss=float(loss.detach()),gradient_norm_before_clip=norm,gradient_tensors=len(parameters),gradient_values=31640742,all_gradients_and_parameters_and_Adam_moments_finite=True,optimizer_step_min=min(steps),optimizer_step_max=max(steps),CUDA_rng_sha256=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest())
     line=(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n').encode();stream.write(line);ledgerhash.update(line)
     r.update(state='running_full_KITTI_source_RD_training',update_count=step+1,last_update=row)
     if (step+1)%64==0:save();print(json.dumps(dict(candidate=args.candidate,updates=step+1,loss=row['loss'])),flush=True)
     del pair,left,right,output,mse,bpp,loss
    path=out/f'checkpoint-epoch{epoch+1}.pt';torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),epoch=epoch+1,updates=(epoch+1)*3712,lambda_RD=value),path)
    r['completed_epochs'].append(dict(epoch=epoch+1,updates=(epoch+1)*3712,path=str(path.relative_to(ROOT)),sha256=sha(path),states=states(model.state_dict())));save()
  assert r['update_count']==37120 and len(r['completed_epochs'])==10
  r.update(sources_after={p:sha(ROOT/p) for p in before},base_freeze_after=freeze(),updates_jsonl_sha256=ledgerhash.hexdigest(),final_states=states(model.state_dict()),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
  assert r['sources_after']==before and r['base_freeze_after']==lock['base_freeze'] and r['peak_reserved_bytes']+2*1024**3<=free
  r['state']='finished_all37120_source_RD_updates_pending_actual_terminal_full_audit_and_real_byte_calibration'
 except BaseException:r.update(state='failed',traceback=traceback.format_exc());raise
 finally:r['finished_unix']=time.time();save()
if __name__=='__main__':main()
