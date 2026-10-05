"""Whole native/local records and independent FP64 Adam arithmetic audit."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
DIR=ROOT/'data/engineering/cost-field-GPU-optimizer-003'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def desc(a):return dict(dtype=str(a.dtype),shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest())
def states(s):return {k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest()) for k,v in s.items()}
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['native','local'],required=True);args=parser.parse_args()
 torch.set_num_threads(2)
 r=json.loads((DIR/'report.json').read_text());job=str(r['job_id'])
 assert r['state']=='passed_native_received_only_cost_field32_task_optimizer_updates_pending_actual_terminal_and_local_proof'
 if args.mode=='native':
  queue=subprocess.check_output(['squeue','-h','-u','wc296','-o','%i %T %N'],text=True)
  queue='\n'.join(x for x in queue.splitlines() if x.split()[0]==job)
  accounting=subprocess.check_output(['sacct','-j',job,'--noheader','--parsable2','--format=JobID,State,ExitCode'],text=True)
  assert not queue.strip() and f'{job}|COMPLETED|0:0' in accounting and f'{job}.0|COMPLETED|0:0' in accounting
 else:
  terminal=json.loads((DIR/'terminal.json').read_text());assert terminal['actual_terminal'] and terminal['job_id']==job and terminal['report_sha256']==sha(DIR/'report.json')
  accounting,queue=terminal['sacct'],terminal['squeue']
  for p,v in terminal['artifacts_sha256'].items():assert sha(ROOT/p)==v,p
 lock=json.loads((ROOT/'experiments/cost-field/GPU-inputs-003.json').read_text())
 assert r['input_sha256']==sha(ROOT/'experiments/cost-field/GPU-inputs-003.json')
 assert r['sources_before']==r['sources_after']==lock['frozen_inputs'] and r['base_freeze_before']==r['base_freeze_after']==lock['base_freeze']
 assert r['sensor_inputs']==r['sensor_inputs_after']==lock['sensor_inputs']
 if args.mode=='native':
  for p,v in lock['frozen_inputs'].items():assert sha(ROOT/p)==v,p
 assert r['all_parameters_frozen'] and r['all_parameter_gradients_absent'] and r['optimizer_used']
 assert r['labels_loss_side_only'] and not r['GT_or_labels_or_LiDAR_at_forward_entry']
 raw=torch.load(ROOT/'assets/liga-native-001/liga-author-download',map_location='cpu',weights_only=True)['model_state']
 for k in r['sparse_layout_converted']:raw[k]=torch.from_numpy(raw[k].numpy().transpose(4,0,1,2,3).copy())
 author=states(raw);assert len(author)==484 and author==r['loaded_states']==r['final_states']
 assert r['codec_parameters_per_arm']==dict(G=1650,P=1650,S=1650,B=1751)
 initial=torch.load(DIR/'codec-initial.pt',map_location='cpu',weights_only=True)
 assert sha(DIR/'codec-initial.pt')==r['codec_initial_checkpoint_sha256']
 previous={arm:states(s) for arm,s in initial.items()};assert previous==r['codec_initial_states']
 coordinate=r['depth_coordinate_contract'];axis=np.asarray(coordinate['voxel_query_axis'])
 assert len(axis)==72 and abs(axis[0]-2)<1e-6 and abs(axis[-1]-59.6)<4e-6
 for arm in initial:assert np.array_equal(initial[arm]['depth_axis'].numpy(),axis.astype(np.float32))
 conditions=[('identity',10),('identity',10),('awgn',6),('awgn',10),('awgn',18),('rayleigh',6),('rayleigh',10),('rayleigh',18)]
 assert [(x['arm'],x['step'],x['channel'],x['snr_db'],x['frame_id']) for x in r['frames']]==[(arm,step,c,s,('000000','000003')[step%2]) for arm in ('G','P','S','B') for step,(c,s) in enumerate(conditions)]
 adam={};values=uses=gradient_values=0;rows=[];artifacts={};eps=np.finfo(np.float32).eps
 for row in r['frames']:
  arm=row['arm'];step=row['step'];p=ROOT/row['artifact'];assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes'];artifacts[row['artifact']]=sha(p)
  with np.load(p,allow_pickle=False) as f:a={k:f[k] for k in f.files}
  assert set(a)==set(row['arrays'])
  for k,v in a.items():assert np.isfinite(v).all() and desc(v)==row['arrays'][k],k
  values+=sum(v.size for v in a.values())
  assert row['readonly_states']==author and row['all_parameter_gradients_absent'] and row['cut_calls']==1 and row['forbidden_calls']==0
  assert row['codec_states_before']==previous[arm]
  assert row['calls']==dict(detector=1,image_backbone=2,feature_neck=2,build_cost=1,head3D=1,depth_logits=1,forbidden=0)
  assert set(row['inference_keys'])=={'batch_size','left_img','right_img','calib','image_shape','frame_id'}
  checkpoint=ROOT/row['optimizer_checkpoint'];assert sha(checkpoint)==row['optimizer_checkpoint_sha256'];artifacts[row['optimizer_checkpoint']]=sha(checkpoint)
  state=torch.load(checkpoint,map_location='cpu',weights_only=True);assert states(state['codec'])==row['codec_states_after']==row['codec_states'][arm]
  changed=[k for k in previous[arm] if previous[arm][k]!=row['codec_states_after'][k]];assert changed==row['changed_states'] and changed
  previous[arm]=row['codec_states_after']
  names=[k.removeprefix('codec_gradient_') for k in a if k.startswith('codec_gradient_')]
  assert set(names)==set(initial[arm])-{'depth_axis'}
  optimizer=state['optimizer'];group=optimizer['param_groups'];assert len(group)==1
  group=group[0];assert group['lr']==1e-3 and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==0
  assert len(group['params'])==len(names) and sum(a['codec_gradient_'+k].size for k in names)==r['codec_parameters_per_arm'][arm]
  squared=sum(np.square(a['codec_gradient_'+k].astype(np.float64)).sum() for k in names)
  independent_norm=float(np.sqrt(squared));assert abs(independent_norm-row['gradient_norm_before_clip'])<=64*eps*(1+independent_norm)
  coefficient=min(1.,10/(row['gradient_norm_before_clip']+1e-6));max_parameter_error=0.
  for index,name in zip(group['params'],names):
   g=a['codec_gradient_'+name];clipped=a['clipped_gradient_'+name]
   assert np.allclose(clipped,g.astype(np.float64)*coefficient,atol=eps,rtol=8*eps)
   assert np.isfinite(clipped).all() and list(g.shape)==previous[arm][name]['shape']
   gradient_values+=g.size+clipped.size
   key=(arm,name)
   if key not in adam:adam[key]=[np.zeros_like(clipped,dtype=np.float64),np.zeros_like(clipped,dtype=np.float64),initial[arm][name].numpy().astype(np.float64)]
   m,v,parameter=adam[key];gg=clipped.astype(np.float64)
   m=.9*m+.1*gg;v=.999*v+.001*gg*gg;t=step+1
   parameter=parameter-1e-3/(1-.9**t)*m/(np.sqrt(v/(1-.999**t))+1e-8)
   saved=state['codec'][name].numpy().astype(np.float64);error=np.abs(saved-parameter)
   # Fixed conservative FP32-operation envelope, not bit-equivalence or a
   # tolerance chosen from observed errors. Independent recursion uses FP64.
   assert (error<=128*eps*(1+np.abs(parameter))).all(),name
   moment=optimizer['state'][index];assert int(moment['step'])==t
   assert np.allclose(moment['exp_avg'].numpy(),m,rtol=128*eps,atol=128*eps*np.max(np.abs(gg))+eps**2)
   assert np.allclose(moment['exp_avg_sq'].numpy(),v,rtol=128*eps,atol=128*eps*np.max(gg*gg)+eps**2)
   max_parameter_error=max(max_parameter_error,float(error.max()));adam[key]=[m,v,parameter]
  tx=a['wire_tx'];rx=a['wire_received'];assert tx.shape==rx.shape==(1,29640,2)
  energy=float(np.square(tx.astype(np.float64)).sum());assert abs(energy-row['energy'])<1e-8 and abs(energy-29640)<29640e-6
  assert row['complex_uses']==29640 and row['received_only_decode_identical_after_private_clear'];uses+=29640
  if row['channel']=='identity':assert np.array_equal(tx,rx)
  loss=a['losses'];assert (loss>=0).all() and loss[0]==row['cls_stats']['rpn_loss_cls']
  box=np.float32(row['box_stats']['rpn_loss_loc'])
  for k in ('rpn_loss_iou','rpn_loss_dir'):box=np.float32(box+np.float32(row['box_stats'][k]))
  assert loss[1]==float(box) and loss[2]==float(np.float32(np.float32(loss[0])+box))
  assert int((a['assigned_box_cls_labels']>0).sum())==row['positive_anchors']>0
  rows.append(dict(arm=arm,step=step,frame=row['frame_id'],channel=row['channel'],snr_db=row['snr_db'],losses=loss.tolist(),gradient_norm=independent_norm,Adam_FP64_max_parameter_error=max_parameter_error,changed_state_count=len(changed)))
 assert previous==r['codec_final_states'] and all(previous[arm]!=r['codec_initial_states'][arm] for arm in previous)
 for p in (DIR/'inputs').iterdir():artifacts[str(p.relative_to(ROOT))]=sha(p)
 for p,item in lock['sensor_inputs'].items():
  folder=Path(p).parent.name;suffix={'image_2':'left','image_3':'right','calib':'calib','label_2':'label'}[folder]
  copy=DIR/'inputs'/(Path(p).stem+'-'+suffix+Path(p).suffix);assert sha(copy)==item['sha256'] and copy.stat().st_size==item['bytes']
 for p in (DIR/'report.json',DIR/'codec-initial.pt'):artifacts[str(p.relative_to(ROOT))]=sha(p)
 result=dict(state='closed_actual_native32_codec_task_optimizer_updates' if args.mode=='native' else 'passed_all_transferred32_codec_optimizer_records_and_independent_Adam_arithmetic',actual_terminal=True,job_id=job,checked_unix=time.time(),sacct=accounting,squeue=queue,report_sha256=sha(DIR/'report.json'),complete_saved_array_values=int(values),all_gradient_values=int(gradient_values),complex_uses=uses,updates=32,rows=rows,artifacts_sha256=artifacts,verifier_sha256=sha(__file__),limitation='Two training frames/eight engineering updates per arm. No full training, validation/AP, generic superiority, multi-seed effect or numerical gradient oracle. Full decoded tensors are not saved/replayed locally.')
 out=DIR/'terminal.json' if args.mode=='native' else ROOT/'data/provenance/cost-field-optimizer-local-verification-003.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ('state','job_id','updates','complete_saved_array_values','all_gradient_values','complex_uses')}))
if __name__=='__main__':main()
