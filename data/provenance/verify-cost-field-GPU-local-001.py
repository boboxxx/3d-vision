"""Full transferred engineering evidence; independent state, target and wire checks."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / 'data/engineering/cost-field-GPU-engineering-001'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def desc(a):
 return dict(dtype=str(a.dtype),shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest())
def states(s):
 return {k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest()) for k,v in s.items()}
def main():
 torch.set_num_threads(2)
 r=json.loads((DIR/'report.json').read_text());t=json.loads((DIR/'terminal.json').read_text())
 lock=json.loads((ROOT/'experiments/cost-field/GPU-inputs-001.json').read_text())
 assert t['actual_terminal'] and t['job_id']==r['job_id']=='11424889'
 assert '11424889|COMPLETED|0:0' in t['sacct'] and '11424889.0|COMPLETED|0:0' in t['sacct'] and not t['squeue'].strip()
 assert t['report_sha256']==sha(DIR/'report.json')
 for p,v in t['artifacts_sha256'].items():assert sha(ROOT/p)==v,p
 assert r['sources_before']==r['sources_after']==lock['frozen_inputs']
 assert r['base_freeze_before']==r['base_freeze_after']==lock['base_freeze']
 assert r['sensor_inputs']==r['sensor_inputs_after']==lock['sensor_inputs']
 assert r['input_sha256']==sha(ROOT/'experiments/cost-field/GPU-inputs-001.json')
 assert r['all_parameters_frozen'] and r['all_parameter_gradients_absent'] and not r['optimizer_used']
 assert r['labels_loss_side_only'] and not r['GT_or_labels_or_LiDAR_at_forward_entry']
 raw=torch.load(ROOT/'assets/liga-native-001/liga-author-download',map_location='cpu',weights_only=True)['model_state']
 for k in r['sparse_layout_converted']:
  assert k.startswith('lidar_model.backbone_3d.') and raw[k].ndim==5
  raw[k]=torch.from_numpy(raw[k].numpy().transpose(4,0,1,2,3).copy())
 author=states(raw);assert len(author)==484 and author==r['loaded_states']==r['final_states']
 initial=torch.load(DIR/'codec-initial.pt',map_location='cpu',weights_only=True)
 codec=states(initial);assert sum(v.numel() for k,v in initial.items() if k!='depth_axis')==1650
 assert sha(DIR/'codec-initial.pt')==r['codec_initial_checkpoint_sha256']
 assert all(x==codec for x in r['codec_initial_states'].values()) and r['codec_initial_states']==r['codec_final_states']
 assert len(r['frames'])==14
 rows=[];values=uses=gradients=0;noise={};eps=np.finfo(np.float32).eps
 for row in r['frames']:
  frame=row['frame_id'];p=ROOT/row['artifact'];assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes']
  with np.load(p,allow_pickle=False) as f:a={k:f[k] for k in f.files}
  assert set(a)==set(row['arrays'])
  for k,v in a.items():assert np.isfinite(v).all() and desc(v)==row['arrays'][k],k
  values+=sum(v.size for v in a.values())
  assert row['readonly_states']==author and row['codec_states']==r['codec_initial_states']
  assert row['cut_calls']==1 and row['forbidden_calls']==0 and row['all_parameter_gradients_absent']
  assert row['calls']==dict(detector=1,image_backbone=2,feature_neck=2,build_cost=1,head3D=1,depth_logits=1,forbidden=0)
  assert set(row['inference_keys'])=={'batch_size','left_img','right_img','calib','image_shape','frame_id'}
  # This prior full proof independently derives camera/pseudo-LiDAR GT, range
  # corners, preprocessing and public calibration from the original files.
  proof=json.loads((ROOT/'data/provenance/artemis-detector-gradient-local-verification-002.json').read_text())
  assert proof['state']=='passed_complete_actual_frozen_detector_detection_loss_RGB_autograd_evidence'
  oldrow=next(x for x in json.loads((ROOT/'data/engineering/artemis-detector-gradient-GPU-001/report.json').read_text())['frames'] if x['frame_id']==frame)
  oldpath=ROOT/oldrow['artifact'];assert sha(oldpath)==oldrow['sha256']
  with np.load(oldpath,allow_pickle=False) as old:
   for k in ('original_P2','original_P3','image_shape','cropped_P2','cropped_P3','crop_offsets','camera_gt','range_keep_mask','gt_boxes','assigned_box_cls_labels','assigned_box_reg_targets','assigned_reg_weights','assigned_gt_inds'):
    assert np.array_equal(a[k],old[k]),k
  for suffix,folder,ext in [('left','image_2','png'),('right','image_3','png'),('calib','calib','txt'),('label','label_2','txt')]:
   source=DIR/'inputs'/f'{frame}-{suffix}.{ext}'
   assert sha(source)==lock['sensor_inputs'][f'data/kitti/training/{folder}/{frame}.{ext}']['sha256']
  losses=a['losses'];assert losses.shape==(3,) and (losses>=0).all()
  box=np.float32(row['box_stats']['rpn_loss_loc'])
  for k in ('rpn_loss_iou','rpn_loss_dir'):box=np.float32(box+np.float32(row['box_stats'][k]))
  assert losses[0]==row['cls_stats']['rpn_loss_cls'] and losses[1]==float(box)
  assert losses[2]==float(np.float32(np.float32(losses[0])+box))
  summary=dict(frame=frame,arm=row['arm'],channel=row['channel'],losses=losses.tolist(),all_values=sum(v.size for v in a.values()))
  if row['arm']!='native_identity':
   tx=a['wire_tx'];rx=a['wire_received'];assert tx.shape==rx.shape==(1,29640,2)
   packet=tx.astype(np.float64).reshape(1,20,78,19,2)
   energy_errors=[float(np.max(np.abs(np.square(packet[...,:3,:]).sum(-1)-1)))]
   for low in (3,7,11,15):energy_errors.append(float(np.max(np.abs(np.square(packet[...,low:low+4,:]).sum((-1,-2))-4))))
   assert max(energy_errors)<32*eps
   energy=float(np.square(tx.astype(np.float64)).sum());assert energy==row['energy'] and abs(energy-29640)<29640e-6
   assert row['complex_uses']==29640 and row['received_only_decode_identical_after_private_clear'];uses+=29640
   if row['channel']=='identity':assert np.array_equal(tx,rx)
   else:
    n=rx.astype(np.float64)-tx.astype(np.float64)
    if frame in noise:
     reference,refbound=noise[frame]
     bound=8*eps*(1+np.abs(rx)+np.abs(tx))+refbound
     assert (np.abs(n-reference)<=bound).all()
    else:noise[frame]=(n,8*eps*(1+np.abs(rx)+np.abs(tx)))
    # Predefined six-standard-error checks, rather than a fitted variance range.
    assert abs(n.mean())<=6*np.sqrt(.05/n.size)
    assert abs(np.mean(n*n)-.05)<=6*.05*np.sqrt(2/n.size)
    summary['AWGN_real_mean']=float(n.mean());summary['AWGN_real_second_moment']=float(np.mean(n*n))
   gradient_keys=[k for k in a if k.startswith('codec_gradient_')]
   assert len(gradient_keys)==13
   for k in gradient_keys:
    name=k.removeprefix('codec_gradient_');assert list(a[k].shape)==codec[name]['shape']
    assert int(np.count_nonzero(a[k]))==row['codec_gradient_nonzero'][k];gradients+=a[k].size
   for k in ('cost_encoder','cost_decoder','appearance_encoder','appearance_decoder'):assert np.count_nonzero(a['codec_gradient_'+k+'.weight'])>0
   assert row['decoded_cost']['shape']==[1,32,72,80,312] and row['decoded_appearance']['shape']==[1,32,80,312]
   summary['group_energy_max_errors']=energy_errors
  rows.append(summary)
 assert values==t['complete_saved_array_values']==74988427 and uses==355680
 result=dict(state='passed_all_saved_native_cost_field14_case_arrays_states_targets_wires_and_gradients',actual_terminal=True,job_id=r['job_id'],checked_unix=time.time(),verifier_sha256=sha(__file__),whole_array_values=int(values),all_codec_gradient_values=int(gradients),complex_uses=uses,rows=rows,
  report_sha256=sha(DIR/'report.json'),terminal_sha256=sha(DIR/'terminal.json'),limitation='Untrained engineering; whole decoded feature fields not saved or independently replayed locally. Normalized RBF cancellation is unresolved in this retained version. No new AP, formal training, geometry benefit or finite-difference gradient oracle.')
 out=ROOT/'data/provenance/cost-field-GPU-local-verification-001.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ('state','whole_array_values','all_codec_gradient_values','complex_uses')}))
if __name__=='__main__':main()
