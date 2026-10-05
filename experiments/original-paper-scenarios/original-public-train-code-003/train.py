"""Full public split; frozen original semantic arithmetic, safe exclusive outputs."""
import argparse,hashlib,io,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'reproduction/cao2025'))
from model import SemanticVariant
from wireless import WirelessVariant
from stages import EPOCHS,configure,forward_loss,frozen_state_keys,optimizer_groups,phase

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def digest_state(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def admission():
 lock=read(HERE/'inputs.json');assert all(sha(ROOT/p)==v for p,v in lock['files'].items());assert sha(ROOT/lock['spynet_path'])==lock['spynet_sha256']
 directory=ROOT/'data/engineering/original-full-train-ROI-002';native=read(directory/'native-audit.json');local=read(ROOT/'data/provenance/original-full-train-ROI-local-audit-002.json')
 assert native['state']=='passed_all3712_preserved_rows_7424_native_RGB_sources_and_masks' and local['state']=='passed_all3712_transferred_preserved_rows_7424_mask_and_native_source_metadata'
 assert local['native_audit_sha256']==sha(directory/'native-audit.json') and native['records_sha256']==local['records_sha256']==sha(directory/'records.jsonl')
 old=read(ROOT/'data/engineering/artemis-original-native-001.json');terminal=read(ROOT/'data/engineering/artemis-original-native-001-terminal.json')
 assert old['state']=='passed_original_native_PRO6000_GPU_engineering' and terminal['state'].startswith('completed')
 return lock,directory

def main():
 p=argparse.ArgumentParser();p.add_argument('--scope',choices=['engineering','formal'],required=True);p.add_argument('--stage',type=int,choices=[1,2,3,4],required=True);p.add_argument('--predecessor-report',type=Path);p.add_argument('--predecessor-audit',type=Path);p.add_argument('--gate',type=Path);a=p.parse_args()
 assert a.scope!='engineering' or a.stage==1
 out=ROOT/'data/runs'/f'original-public30-{a.scope}-seed17-003-stage{a.stage}';assert not out.exists();out.mkdir(parents=True)
 r=dict(state='starting',scope=a.scope,stage=a.stage,seed=17,job_id=os.environ.get('SLURM_JOB_ID'),update_count=0,completed_epochs=[],started_unix=time.time(),training_frames=3712,nominal_rate=30,GT_calibration_LiDAR_validation_read=False,independent_raw_gradient_Adam_replay=False)
 def save():
  temp=out/'report.tmp';temp.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');os.replace(temp,out/'report.json')
 save()
 try:
  lock,directory=admission();r['input_sha256']=sha(HERE/'inputs.json');r['sources_before']=lock['files']
  if a.scope=='formal':
   assert a.gate is not None;gate=read(a.gate);assert gate['state']=='closed_actual_GPU_native_local_original_public_engineering003' and gate['input_sha256']==r['input_sha256'];r['engineering_gate_sha256']=sha(a.gate)
  manifest=read(ROOT/'data/provenance/cost-field-full-dataset-manifest-004.json');ids=manifest['splits']['train']['ids'];rows=[json.loads(line) for line in (directory/'records.jsonl').read_text().splitlines()];assert [v['frame_id'] for v in rows]==ids;byid={v['frame_id']:v for v in rows}
  allowed={str((ROOT/f'data/kitti/training/{camera}/{frame}.png').resolve()) for frame in ids for camera in ('image_2','image_3')}
  def barrier(event,args):
   if event=='open' and isinstance(args[0],(str,bytes)):
    name=os.fsdecode(args[0]);mode,flags=args[1:3];reading=(isinstance(mode,str) and 'r' in mode) or (mode is None and flags&os.O_ACCMODE==os.O_RDONLY)
    if reading and ('/data/kitti/' in name or any(s in name for s in ('/label_2/','/velodyne/','/calib/'))):assert str(Path(name).resolve()) in allowed,'Forbidden non-training/non-RGB sensor read'
  sys.addaudithook(barrier);torch.set_num_threads(2);torch.manual_seed(17);np.random.seed(17);torch.cuda.manual_seed_all(17)
  assert torch.__version__=='2.7.1+cu128' and np.__version__=='1.26.4' and torch.cuda.device_count()==1 and torch.cuda.get_device_capability()==(12,0)
  name=torch.cuda.get_device_name();assert 'RTX PRO 6000' in name
  model=WirelessVariant(SemanticVariant(ROOT/lock['spynet_path'])).cuda();parameters=dict(model.named_parameters());assert len(model.state_dict())==768 and len(parameters)==718 and sum(x.numel() for x in parameters.values())==12970798
  if a.stage>1:
   assert a.predecessor_report is not None and a.predecessor_audit is not None;parent=read(a.predecessor_report);proof=read(a.predecessor_audit)
   assert parent['state']=='finished_complete_public_semantic_stage_pending_actual_terminal_and_audit' and parent['scope']=='formal' and parent['stage']==a.stage-1 and proof['state']=='passed_complete_original_public_semantic_stage' and proof['report_sha256']==sha(a.predecessor_report) and proof['actual_terminal_closed']
   checkpoint=ROOT/parent['completed_epochs'][-1]['path'];assert sha(checkpoint)==parent['completed_epochs'][-1]['sha256'];previous=torch.load(checkpoint,map_location='cpu',weights_only=True);model.load_state_dict(previous['model_state'],strict=True)
   r.update(predecessor_report_sha256=sha(a.predecessor_report),predecessor_audit_sha256=sha(a.predecessor_audit),predecessor_checkpoint_sha256=sha(checkpoint));del previous
  initial=out/'initial.pt';torch.save(dict(model_state={k:v.detach().cpu() for k,v in model.state_dict().items()}),initial);r.update(initial_sha256=sha(initial),GPU=name,capability=[12,0],source_guard=True)
  optimizer=torch.optim.Adam(optimizer_groups(model,a.stage),betas=(.9,.999),eps=1e-8,weight_decay=0.);counts={k:0 for k in parameters};epochs=1 if a.scope=='engineering' else EPOCHS[a.stage]
  with (out/'updates.jsonl').open('x') as stream:
   for epoch in range(1,epochs+1):
    assert all(sha(ROOT/p)==v for p,v in lock['files'].items());current=phase(a.stage,epoch);active=configure(model,current);frozen=frozen_state_keys(model);before={k:digest_state(model.state_dict()[k]) for k in frozen}
    order=ids[:3] if a.scope=='engineering' else np.random.default_rng(17+1000*a.stage+epoch).permutation(ids).tolist()
    for index,frame in enumerate(order):
     row=byid[frame];images=[];source={}
     for view in row['views']:
      rel=f"data/kitti/training/{view['camera']}/{frame}.png";blob=(ROOT/rel).read_bytes();meta=manifest['files'][rel];assert len(blob)==meta['bytes'] and hashlib.sha256(blob).hexdigest()==view['image_sha256']==meta['sha256'];source[rel]=meta['sha256']
      with Image.open(io.BytesIO(blob)) as im:assert im.mode=='RGB' and [im.height,im.width]==view['shape'];rgb=np.asarray(im).copy()
      images.append(torch.from_numpy(rgb).permute(2,0,1)[None].cuda().float()/255.)
     model.zero_grad(set_to_none=True);result=forward_loss(model,*images,[v['boxes'] for v in row['views']],current);loss=result['loss'];assert result['erasure'] is None and result['accounting'] is None and torch.isfinite(loss);loss.backward()
     grad={k for k,v in parameters.items() if v.grad is not None};assert grad==active and torch.stack([torch.isfinite(parameters[k].grad).all() for k in grad]).all()
     norm=float(torch.nn.utils.clip_grad_norm_([parameters[k] for k in grad],10.,error_if_nonfinite=True));optimizer.step()
     for k in grad:counts[k]+=1
     assert torch.stack([torch.isfinite(v).all() for v in model.state_dict().values()]).all() and torch.stack([torch.isfinite(s[k]).all() for s in optimizer.state.values() for k in ['exp_avg','exp_avg_sq']]).all()
     r['update_count']+=1;record=dict(step=r['update_count'],epoch=epoch,index=index,frame_id=frame,shape=list(images[0].shape),box_counts=[len(v['boxes']) for v in row['views']],source_png_sha256=source,loss_type=current['loss'],loss=float(loss.detach()),gradient_parameter_tensors=len(grad),preclip_grad_norm=norm,all_gradients_parameters_moments_finite=True)
     stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();r.update(state='running_complete_public_semantic_stage',last_update=record)
     if r['update_count']%64==0:save();print(json.dumps(record),flush=True)
     del images,result,loss
    assert all(digest_state(model.state_dict()[k])==v for k,v in before.items());assert all(sha(ROOT/p)==v for p,v in lock['files'].items())
    checkpoint=out/f'checkpoint-epoch{epoch}.pt';torch.save(dict(model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer_state=optimizer.state_dict(),stage=a.stage,epoch=epoch,updates=r['update_count'],parameter_update_counts=counts.copy(),CPU_rng=torch.get_rng_state(),CUDA_rng=torch.cuda.get_rng_state_all()),checkpoint)
    r['completed_epochs'].append(dict(epoch=epoch,phase=current,samples=len(order),updates=r['update_count'],path=str(checkpoint.relative_to(ROOT)),sha256=sha(checkpoint),frozen_states_identical=True,frozen_state_tensors=len(frozen),parameter_update_counts=counts.copy()));save()
  assert r['update_count']==(3 if a.scope=='engineering' else 3712*EPOCHS[a.stage]);r.update(state='finished_complete_public_semantic_stage_pending_actual_terminal_and_audit',records_sha256=sha(out/'updates.jsonl'),sources_after={p:sha(ROOT/p) for p in lock['files']},peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
 except BaseException:r.update(state='failed',traceback=traceback.format_exc());raise
 finally:r['finished_unix']=time.time();save()
if __name__=='__main__':main()
