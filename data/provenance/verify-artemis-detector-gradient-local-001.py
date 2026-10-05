"""All transferred gradients, safe full states and independent loss-side GT geometry."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
DIR=ROOT/'data/engineering/artemis-detector-gradient-GPU-001'
FORWARD=ROOT/'data/engineering/artemis-full-detector-GPU-001'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def descriptor(a):
 return dict(dtype=str(a.dtype),shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest())
def main():
 torch.set_num_threads(2)
 r=json.loads((DIR/'report.json').read_text());terminal=json.loads((DIR/'terminal.json').read_text())
 assert terminal['actual_terminal'] and terminal['job_id']=='11424856'
 assert terminal['report_sha256']==sha(DIR/'report.json')
 assert '11424856|COMPLETED|0:0' in terminal['sacct'] and not terminal['squeue'].strip()
 assert r['state']=='passed_native_detection_loss_RGB_backward_pending_actual_terminal_and_local_proof'
 assert r['GPU']=='NVIDIA RTX PRO 6000 Blackwell Server Edition' and r['capability']==[12,0]
 assert r['all_parameters_frozen'] and r['all_parameter_gradients_absent'] and not r['optimizer_used']
 assert not r['GT_or_labels_or_LiDAR_at_forward_entry'] and r['labels_loss_side_only']
 lock=json.loads((ROOT/'experiments/artemis-detector-framework/gradient-inputs-001.json').read_text())
 assert r['input_sha256']==sha(ROOT/'experiments/artemis-detector-framework/gradient-inputs-001.json')
 assert r['sources_before']==r['sources_after']==lock['frozen_inputs']
 assert r['base_freeze_before']==r['base_freeze_after']==lock['base_freeze']
 assert r['sensor_inputs']==r['sensor_inputs_after']==lock['sensor_inputs']
 checkpoint=ROOT/'assets/liga-native-001/liga-author-download'
 assert sha(checkpoint)=='3f36af30b686f23d0868482675177b29cc50c2b378577dd12f9092ddb0235b2e'
 raw=torch.load(checkpoint,map_location='cpu',weights_only=True)['model_state'];assert len(raw)==484
 expected={}
 for k,v in raw.items():
  assert torch.isfinite(v).all()
  if k in r['sparse_layout_converted']:
   assert k.startswith('lidar_model.backbone_3d.') and k.endswith('.weight') and v.ndim==5
   v=torch.from_numpy(v.numpy().transpose(4,0,1,2,3).copy())
  expected[k]=dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest())
 assert expected==r['loaded_states']==r['final_states']
 audit=json.loads((ROOT/'data/provenance/artemis-full-detector-GPU-local-audit-003.json').read_text())
 assert len(audit['frames'])==2 and audit['all_input_RGB_values_verified']==4792320
 results=[];total=0;all_gradient=0
 for row in r['frames']:
  frame=row['frame_id'];p=ROOT/row['artifact'];assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes']
  with np.load(p,allow_pickle=False) as f:a={k:f[k] for k in f.files}
  assert set(a)==set(row['arrays'])
  for k,v in a.items():assert descriptor(v)==row['arrays'][k] and np.isfinite(v).all()
  total+=sum(v.size for v in a.values())
  assert row['readonly_states']==expected and row['all_parameter_gradients_absent'] and row['forbidden_calls']==0
  assert row['calls']==dict(detector=1,image_backbone=2,feature_neck=2,build_cost=1,head3D=1,depth_logits=1,forbidden=0)
  assert set(row['inference_keys'])=={'batch_size','left_img','right_img','calib','image_shape','frame_id'}
  # Independently audited original forward inputs and native predictions are unchanged.
  with np.load(FORWARD/(frame+'.npz'),allow_pickle=False) as f:
   for k in ('left_input','right_input','image_shape','original_P2','original_P3','cropped_P2','cropped_P3','crop_offsets','pred_boxes','pred_scores','pred_labels'):
    assert np.array_equal(a[k],f[k]),k
  for view,folder in [('left','image_2'),('right','image_3')]:
   source=DIR/'inputs'/(frame+'-'+view+'.png')
   assert sha(source)==lock['sensor_inputs']['data/kitti/training/'+folder+'/'+frame+'.png']['sha256']
   grad=a[view+'_gradient'];assert grad.dtype==np.float32 and grad.shape==a[view+'_input'].shape
   assert int(np.count_nonzero(grad))==row['gradient_nonzero'][view]>0;all_gradient+=grad.size
  for suffix,folder in [('calib','calib'),('label','label_2')]:
   source=DIR/'inputs'/(frame+'-'+suffix+'.txt')
   assert sha(source)==lock['sensor_inputs']['data/kitti/training/'+folder+'/'+frame+'.txt']['sha256']
  camera=[];classes=[]
  for line in (DIR/'inputs'/(frame+'-label.txt')).read_text().splitlines():
   fields=line.split();cls={'Van':'Car','Person_sitting':'Pedestrian'}.get(fields[0],fields[0])
   if cls not in ['Car','Pedestrian','Cyclist']:continue
   camera.append([*map(float,fields[11:14]),float(fields[10]),float(fields[8]),float(fields[9]),float(fields[14])])
   classes.append(['Car','Pedestrian','Cyclist'].index(cls)+1)
  camera=np.asarray(camera,np.float32).reshape(-1,7);assert np.array_equal(camera,a['camera_gt'])
  # Independent pseudo-LiDAR coordinates: (camera Z,-X,-Y + H/2,L,W,H,-RY-pi/2).
  boxes=np.column_stack([camera[:,2],-camera[:,0],-camera[:,1]+camera[:,4]/2,camera[:,3],camera[:,5],camera[:,4],-(camera[:,6]+np.float32(np.pi/2))])
  signs=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)],np.float64)/2
  corners=boxes[:,None,3:6].astype(np.float64)*signs[None]
  angle=boxes[:,6].astype(np.float64);c=np.cos(angle)[:,None];s=np.sin(angle)[:,None]
  x=corners[:,:,0]*c-corners[:,:,1]*s;y=corners[:,:,0]*s+corners[:,:,1]*c
  corners=np.stack([x,y,corners[:,:,2]],-1)+boxes[:,None,:3]
  limits=np.array([2,-30.4,-3,59.6,30.4,1]);mask=((corners>=limits[:3])&(corners<=limits[3:])).all(-1).any(-1)
  assert np.array_equal(mask,a['range_keep_mask'])
  gt=np.column_stack([boxes[mask],np.asarray(classes,np.float32)[mask]])
  error=float(np.max(np.abs(gt-a['gt_boxes'])));assert error<=2e-6
  cls=a['assigned_box_cls_labels'];pos=cls>0;assert int(pos.sum())==row['positive_anchors']>0
  assert np.isin(cls,[-1,0,1,2,3]).all() and np.isin(cls[pos],gt[:,7]).all()
  assert a['assigned_box_reg_targets'].shape==(*cls.shape,7)
  assert a['assigned_reg_weights'].shape==a['assigned_gt_inds'].shape==cls.shape
  loss=a['losses'];assert loss.shape==(3,) and (loss>=0).all()
  assert loss[0]==row['cls_stats']['rpn_loss_cls']
  components=row['box_stats'];box_sum=sum(components[k] for k in ('rpn_loss_loc','rpn_loss_iou','rpn_loss_dir'))
  assert abs(box_sum-loss[1])<=1e-7 and abs(loss[0]+loss[1]-loss[2])<=1e-7
  results.append(dict(frame=frame,whole_array_values=int(sum(v.size for v in a.values())),positive_anchors=int(pos.sum()),
   losses=loss.tolist(),independent_GT_max_error=error,both_gradient_values=int(a['left_gradient'].size+a['right_gradient'].size),
   original_sensor_only_forward_inputs_and_predictions_exact=True,full484_states_unchanged=True))
 assert total==terminal['complete_saved_array_values']
 result=dict(state='passed_complete_actual_frozen_detector_detection_loss_RGB_autograd_evidence',checked_unix=time.time(),
  verifier_sha256=sha(__file__),job_id='11424856',actual_terminal=True,whole_array_values=int(total),all_gradient_values=all_gradient,
  frames=results,artifacts_sha256={str(p.relative_to(ROOT)):sha(p) for p in DIR.rglob('*') if p.is_file()},
  limitation='Full saved autograd evidence and independently transformed GT; no finite-difference oracle, optimized codec, sparse teacher GPU, fullval or geometry-specific AP benefit')
 target=ROOT/'data/provenance/artemis-detector-gradient-local-verification-001.json';assert not target.exists()
 target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('state','whole_array_values','all_gradient_values','frames')}))
if __name__=='__main__':main()
