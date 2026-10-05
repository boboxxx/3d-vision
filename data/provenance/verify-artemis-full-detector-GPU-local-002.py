"""Complete transferred two-frame native forward proof, safe checkpoint and pixels."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[2]
DIR=ROOT/'data/engineering/artemis-full-detector-GPU-001'
CHECKPOINT=ROOT/'assets/liga-native-001/liga-author-download'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def state(v):
 a=v.detach().cpu().contiguous()
 return dict(dtype=str(a.dtype),shape=list(a.shape),sha256=hashlib.sha256(a.numpy().tobytes()).hexdigest())

def calibration(p):
 rows={}
 for line in p.read_text().splitlines():
  if not line.strip():continue
  key,*values=line.split();rows[key.rstrip(':')]=np.array([float(v) for v in values],np.float32)
 return rows['P2'].reshape(3,4),rows['P3'].reshape(3,4)

def main():
 torch.set_num_threads(2)
 r=json.loads((DIR/'report.json').read_text());terminal=json.loads((DIR/'terminal.json').read_text())
 assert r['state']=='passed_full484_native_sensor_only_PRO6000_forward_pending_actual_terminal_and_local_proof'
 assert terminal['actual_terminal'] and terminal['state']=='closed_actual_native_full_detector_GPU_terminal'
 assert terminal['report_sha256']==sha(DIR/'report.json')
 assert r['GPU']=='NVIDIA RTX PRO 6000 Blackwell Server Edition' and r['capability']==[12,0]
 assert r['torch_version']=='2.7.1+cu128' and r['numpy_version']=='1.26.4' and r['mmcv_version']=='1.7.2'
 assert r['base_runtime_unchanged'] and r['frozen_inputs_unchanged'] and r['sources_before']==r['sources_after']
 lock=json.loads((ROOT/'experiments/artemis-detector-framework/GPU-inputs-001.json').read_text())
 assert r['input_sha256']==sha(ROOT/'experiments/artemis-detector-framework/GPU-inputs-001.json')
 assert r['sources_before']==lock['frozen_inputs'] and r['sensor_inputs']==lock['sensor_inputs']
 assert r['base_freeze_before']==r['base_freeze_after']==lock['base_freeze']
 assert sha(CHECKPOINT)=='3f36af30b686f23d0868482675177b29cc50c2b378577dd12f9092ddb0235b2e' and CHECKPOINT.stat().st_size==93383301
 raw=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)['model_state'];assert len(raw)==484
 converted=r['sparse_layout_converted'];assert len(set(converted))==len(converted)
 expected={}
 for k,v in raw.items():
  assert torch.isfinite(v).all()
  if k in converted:
   assert k.startswith('lidar_model.backbone_3d.') and k.endswith('.weight') and v.ndim==5
   # Independent source RSCK -> KRSC permutation, preserving all tensor bytes.
   a=np.transpose(v.numpy(),(4,0,1,2,3)).copy();v=torch.from_numpy(a)
  expected[k]=state(v)
 assert expected==r['loaded_states']==r['final_states']
 assert sum(k.startswith('lidar_model.') for k in raw)==110
 assert r['state_load']=='complete_strict484_safe_weights_only_true'
 assert r['GT_or_labels_or_LiDAR_read'] is False and r['DDP_backend']=='nccl'
 assert r['initialization_only_overrides']==dict(feature_backbone_pretrained='torchvision://resnet34',teacher_PRETRAINED_MODEL='./ckpt/second_s4_hg.iouloss.ep78.backbone-no-final-bnrelu.input-only-xyz.default-lr-policy-with-wd-decay-78ep.pth')
 artifacts={str((DIR/'report.json').relative_to(ROOT)):sha(DIR/'report.json'),str((DIR/'terminal.json').relative_to(ROOT)):sha(DIR/'terminal.json'),str(CHECKPOINT.relative_to(ROOT)):sha(CHECKPOINT)}
 results=[];pixels=0
 assert [row['frame_id'] for row in r['frames']]==['000000','000003']
 for row in r['frames']:
  frame=row['frame_id'];p=ROOT/row['artifact'];assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes'];artifacts[row['artifact']]=sha(p)
  with np.load(p,allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
  assert set(arrays)==set(row['arrays'])
  for k,a in arrays.items():
   descriptor=dict(dtype=str(a.dtype),shape=list(a.shape),sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
   assert descriptor==row['arrays'][k] and np.isfinite(a).all(),k
  assert row['readonly_states']==expected and row['teacher_or_loss_calls']==0
  assert row['calls']==dict(detector=1,image_backbone=2,feature_neck=2,build_cost=1,head3D=1,depth_logits=1,forbidden=0)
  assert set(row['inference_keys'])=={'batch_size','left_img','right_img','calib','image_shape','frame_id'}
  sources=[];original_shape=None;image_error=0.
  for view,folder in [('left','image_2'),('right','image_3')]:
   p=DIR/'inputs'/(frame+'-'+view+'.png');key='data/kitti/training/'+folder+'/'+frame+'.png';item=lock['sensor_inputs'][key]
   assert sha(p)==item['sha256'] and p.stat().st_size==item['bytes'];artifacts[str(p.relative_to(ROOT))]=sha(p)
   with Image.open(p) as im:rgb=np.asarray(im.convert('RGB')).copy()
   assert rgb.dtype==np.uint8 and rgb.shape[2]==3
   h,w=rgb.shape[:2];crop_h,crop_w=min(h,320),min(w,1280);x=(w-crop_w)//2;y=h-crop_h
   source=(rgb.astype(np.float32)/np.float32(255))*np.float32(255)
   cropped=source[y:y+crop_h,x:x+crop_w]
   normalized=(cropped/np.float32(255)-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
   pad_h=(-crop_h)%32;pad_w=(-crop_w)%32
   expected_img=np.pad(normalized,((0,pad_h),(0,pad_w),(0,0)))[None].transpose(0,3,1,2)
   actual=arrays[view+'_input'];assert actual.dtype==np.float32 and actual.shape==expected_img.shape
   e=float(np.max(np.abs(actual-expected_img)));assert e<=1e-6;e_pad=actual[:,:,crop_h:,:];assert (e_pad==0).all()
   assert (actual[:,:,:,crop_w:]==0).all();pixels+=actual.size;image_error=max(image_error,e)
   if original_shape is None:original_shape=(h,w);offset=(x,y)
   else:assert original_shape==(h,w)
  assert np.array_equal(arrays['image_shape'],np.array([original_shape])) and np.array_equal(arrays['crop_offsets'],offset)
  p=DIR/'inputs'/(frame+'-calib.txt');item=lock['sensor_inputs']['data/kitti/training/calib/'+frame+'.txt']
  assert sha(p)==item['sha256'] and p.stat().st_size==item['bytes'];artifacts[str(p.relative_to(ROOT))]=sha(p)
  p2,p3=calibration(p);assert np.array_equal(arrays['original_P2'],p2) and np.array_equal(arrays['original_P3'],p3)
  # Independent image-coordinate translation; original implementation factors K,T in FP32.
  H=np.eye(3,dtype=np.float64);H[0,2]=-offset[0];H[1,2]=-offset[1]
  cal_errors=[]
  for view,matrix in [('P2',p2),('P3',p3)]:
   err=float(np.max(np.abs(arrays['cropped_'+view]-H@matrix.astype(np.float64))));assert err<=3e-4
   cal_errors.append(err)
  boxes,scores,labels=arrays['pred_boxes'],arrays['pred_scores'],arrays['pred_labels']
  assert boxes.ndim==2 and boxes.shape[1]==7 and scores.shape==labels.shape==(len(boxes),)
  assert len(boxes)==row['prediction_count'] and ((scores>=0)&(scores<=1)).all()
  assert ((labels>=1)&(labels<=3)).all() and np.issubdtype(labels.dtype,np.integer) and (boxes[:,3:6]>0).all()
  rawboxes=arrays['batch_box_preds'];logits=arrays['batch_cls_preds']
  assert rawboxes.ndim==3 and rawboxes.shape[0]==1 and rawboxes.shape[2]==7
  assert logits.shape==(1,rawboxes.shape[1],3)
  membership_error=0.
  for box,score,label in zip(boxes,scores,labels):
   differences=np.max(np.abs(rawboxes[0]-box),-1);indices=np.flatnonzero(differences<=1e-6);assert len(indices)>0
   expected_scores=np.exp(-np.logaddexp(0.,-logits[0,indices,int(label)-1].astype(np.float64)))
   e=float(np.min(np.abs(expected_scores-score)));assert e<=3e-7;membership_error=max(membership_error,e)
  axis=arrays['depth_samples'];low=arrays['depth_axis_low'];dl=arrays['depth_logits_low']
  assert axis.ndim==low.ndim==1 and (np.diff(axis)>0).all() and (np.diff(low)>0).all()
  assert dl.ndim==5 and dl.shape[0:2]==(1,1) and dl.shape[2]==len(axis)
  # Full native depth regression from all logits, independent NumPy softmax/reduction.
  assert arrays['depth_pred'].shape==(dl.shape[0],dl.shape[3],dl.shape[4])
  depth_error=0.;depth_cells=0
  # Every spatial row and every depth atom, with bounded FP64 scratch space.
  for first in range(0,dl.shape[3],4):
   last=min(first+4,dl.shape[3]);value=dl[:,0,:,first:last,:].astype(np.float64)
   maximum=value.max(1,keepdims=True);prob=np.exp(value-maximum);prob/=prob.sum(1,keepdims=True)
   depth=(prob*axis[None,:,None,None]).sum(1)
   depth_error=max(depth_error,float(np.max(np.abs(arrays['depth_pred'][:,first:last,:]-depth))))
   depth_cells+=depth.size
  assert depth_cells==arrays['depth_pred'].size and depth_error<=3e-5
  assert arrays['depth_pred_local'].shape==arrays['depth_pred'].shape and (arrays['depth_pred_local']>=axis[0]-1e-5).all() and (arrays['depth_pred_local']<=axis[-1]+1e-5).all()
  results.append(dict(frame=frame,all_array_values=int(sum(a.size for a in arrays.values())),input_RGB_max_error=image_error,public_calibration_max_errors=cal_errors,
   prediction_count=len(boxes),all_postNMS_rows_present_in_raw_head=True,raw_score_max_error=membership_error,native_depth_regression_max_error=depth_error,
   states_verified=484,saved_array_count=len(arrays)))
 result=dict(state='passed_complete_two_frame_full484_PRO6000_native_forward_local_proof',checked_unix=time.time(),verifier_sha256=sha(__file__),
  artifacts_sha256=artifacts,actual_terminal=True,job_id=r['job_id'],GPU=r['GPU'],capability=r['capability'],full484_states_verified=True,
  all_input_RGB_values_verified=pixels,frames=results,limitation='Two actual sensor-only full-detector GPU forwards; no training/backward/sparse-teacher-GPU or fullval/AP/new-method claim')
 out=ROOT/'data/provenance/artemis-full-detector-GPU-local-verification-002.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ('state','all_input_RGB_values_verified','frames')}))
if __name__=='__main__':main()
