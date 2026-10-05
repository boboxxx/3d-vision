"""Complete strict author detector, two actual sensor-only sm120 forward passes."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
INPUT=HERE.parent/'GPU-inputs-001.json'
OUT=ROOT/'data/engineering/artemis-full-detector-GPU-001'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def state_record(model):
 result={}
 for k,v in model.state_dict().items():
  a=v.detach().cpu().contiguous()
  result[k]=dict(dtype=str(a.dtype),shape=list(a.shape),sha256=hashlib.sha256(a.numpy().tobytes()).hexdigest())
 return result

def freeze():
 e=os.environ.copy();e.pop('PYTHONPATH',None)
 return subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True,env=e)

def main():
 assert not OUT.exists();OUT.mkdir(parents=True)
 lock=json.loads(INPUT.read_text())
 report=dict(state='starting',stage='check_inputs',started_unix=time.time(),job_id=os.environ.get('SLURM_JOB_ID'),
   input_sha256=sha(INPUT),frames=[],GT_or_labels_or_LiDAR_read=False,teacher_GPU_or_backward_verified=False)
 def save():(OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 save();group=None;handles=[]
 try:
  torch.set_num_threads(2);torch.manual_seed(17);np.random.seed(17)
  sources={p:sha(ROOT/p) for p in lock['frozen_inputs']};assert sources==lock['frozen_inputs']
  report['sources_before']=sources;report['base_freeze_before']=freeze();assert report['base_freeze_before']==lock['base_freeze']
  for p,item in lock['sensor_inputs'].items():assert sha(ROOT/p)==item['sha256'] and (ROOT/p).stat().st_size==item['bytes']
  report['sensor_inputs']=lock['sensor_inputs'];save()
  sys.path[:0]=[str(ROOT/'src'),str(ROOT/'third_party/LIGA-Stereo'),str(ROOT/'third_party/mmdetection_kitti'),str(ROOT/'experiments/original-detector-integration')]
  builds=json.loads((ROOT/'data/engineering/artemis-detector-operators-build-001.json').read_text())
  for package,name in [('build_cost_volume','build_cost_volume_cuda'),('iou3d_nms','iou3d_nms_cuda'),('roiaware_pool3d','roiaware_pool3d_cuda')]:
   paths=[ROOT/p for p in builds['binaries'] if Path(p).name.startswith(name+'.')];assert len(paths)==1
   full='liga.ops.'+package+'.'+name;spec=importlib.util.spec_from_file_location(full,paths[0]);m=importlib.util.module_from_spec(spec);sys.modules[full]=m;spec.loader.exec_module(m)
  from easydict import EasyDict
  from liga.config import cfg_from_yaml_file
  from liga.datasets.stereo_dataset_template import StereoDatasetTemplate
  from liga.models import build_network
  from liga.utils.calibration_kitti import Calibration
  from geocomm.compat import adapt_spconv_state
  from geocomm.inference import SENSOR_INPUT_KEYS
  from adapter import liga_preprocess
  import mmcv,torchvision
  assert torch.__version__=='2.7.1+cu128' and np.__version__=='1.26.4'
  assert mmcv.__version__=='1.7.2' and torchvision.__version__=='0.22.1+cu128'
  assert 'artemis-framework-overlay-005/site-packages' in mmcv.__file__
  assert torch.cuda.device_count()==1 and torch.cuda.get_device_capability(0)==(12,0)
  name=torch.cuda.get_device_name(0);assert 'RTX PRO 6000' in name
  report.update(GPU=name,capability=[12,0],torch_version=torch.__version__,numpy_version=np.__version__,mmcv_version=mmcv.__version__,CUDA_visible_device=os.environ.get('CUDA_VISIBLE_DEVICES'))
  os.chdir(ROOT/'third_party/LIGA-Stereo')
  cfg=cfg_from_yaml_file(str(ROOT/'configs/diagnostic/clean_holdout.yaml'),EasyDict())
  report['initialization_only_overrides']={
   'feature_backbone_pretrained':cfg.MODEL.BACKBONE_3D.feature_backbone_pretrained,
   'teacher_PRETRAINED_MODEL':cfg.MODEL.LIDAR_MODEL.PRETRAINED_MODEL}
  cfg.MODEL.BACKBONE_3D.feature_backbone_pretrained=None;cfg.MODEL.LIDAR_MODEL.PRETRAINED_MODEL=None
  dataset=StereoDatasetTemplate(cfg.DATA_CONFIG,cfg.CLASS_NAMES,training=False,root_path=ROOT/'data/kitti')
  dataset.boxes_gt_in_cam2_view=cfg.DATA_CONFIG.BOXES_GT_IN_CAM2_VIEW
  group=tempfile.TemporaryDirectory(prefix='author-liga-DDP-',dir=OUT)
  dist.init_process_group('nccl',init_method='file://'+group.name+'/rank',rank=0,world_size=1)
  report['stage']='build_complete_model';save()
  model=build_network(cfg.MODEL,len(cfg.CLASS_NAMES),dataset).cuda().eval()
  b=model.backbone_3d
  assert all(x is None for x in (b.semantic_link,b.stereo_feature_link,b.student_semantic_link_encoder,model.rgb_semantic_link))
  checkpoint=ROOT/'assets/liga-native-001/liga-author-download'
  assert sha(checkpoint)=='3f36af30b686f23d0868482675177b29cc50c2b378577dd12f9092ddb0235b2e'
  raw=torch.load(checkpoint,map_location='cpu',weights_only=True)['model_state']
  state=adapt_spconv_state(model,raw);actual=model.state_dict()
  assert len(state)==len(actual)==484 and set(state)==set(actual)
  assert all(v.shape==actual[k].shape and v.dtype==actual[k].dtype and torch.isfinite(v).all() for k,v in state.items())
  model.load_state_dict(state,strict=True)
  assert all(torch.equal(v.detach().cpu(),state[k]) for k,v in model.state_dict().items())
  report['loaded_states']=state_record(model);report['state_load']='complete_strict484_safe_weights_only_true'
  report['sparse_layout_converted']=[k for k in raw if raw[k].shape!=state[k].shape]
  assert sum(k.startswith('lidar_model.') for k in state)==110
  del raw,state,actual
  initial=report['loaded_states'];wrapped=torch.nn.parallel.DistributedDataParallel(model,device_ids=[0],broadcast_buffers=False)
  calls={};pending={};captured={}
  def forbidden(*args,**kwargs):calls['forbidden']+=1;raise RuntimeError('teacher or training loss forbidden')
  handles.append(model.lidar_model.register_forward_pre_hook(forbidden))
  model.get_training_loss=forbidden;model.dense_head.get_loss=forbidden;model.depth_loss_head.get_loss=forbidden
  model.dense_head_2d.get_loss=forbidden
  def entry(module,args):
   assert not module.training and not torch.is_grad_enabled() and set(args[0])==set(SENSOR_INPUT_KEYS)
   assert args[0]['left_img'] is pending['images'][0] and args[0]['right_img'] is pending['images'][1]
   calls['detector']+=1
  def backbone(module,args):
   assert args[0] is pending['images'][calls['image_backbone']];calls['image_backbone']+=1
  def count(key):
   def hook(module,args):calls[key]+=1
   return hook
  def logit(module,args,output):
   assert isinstance(output,torch.Tensor) and torch.isfinite(output).all()
   captured['depth_logits_low']=output.detach().cpu().numpy().copy();calls['depth_logits']+=1
  handles.extend([model.register_forward_pre_hook(entry),b.feature_backbone.register_forward_pre_hook(backbone),
   b.feature_neck.register_forward_pre_hook(count('feature_neck')),b.build_cost.register_forward_pre_hook(count('build_cost')),
   model.dense_head.register_forward_pre_hook(count('head3D')),b.pred_stereo[-1].register_forward_hook(logit)])
  torch.cuda.reset_peak_memory_stats();report['stage']='actual_sensor_only_forward';save()
  for frame in ('000000','000003'):
   calls.update(detector=0,image_backbone=0,feature_neck=0,build_cost=0,head3D=0,depth_logits=0,forbidden=0)
   arrays={};images=[];inputs_dir=OUT/'inputs';inputs_dir.mkdir(exist_ok=True)
   for view,folder in [('left','image_2'),('right','image_3')]:
    p=ROOT/'data/kitti/training'/folder/(frame+'.png');copy=inputs_dir/(frame+'-'+view+'.png');shutil.copyfile(p,copy);assert sha(p)==sha(copy)
    with Image.open(p) as im:rgb=np.asarray(im.convert('RGB')).copy()
    images.append(torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).float()/255.)
   p=ROOT/'data/kitti/training/calib'/(frame+'.txt');cp=inputs_dir/(frame+'-calib.txt');shutil.copyfile(p,cp);assert sha(cp)==sha(p)
   cal=Calibration(str(p));arrays.update(original_P2=cal.P2.copy(),original_P3=cal.P3.copy())
   batch=liga_preprocess(images,cal,frame,dataset.data_augmentor, dataset.collate_batch,device='cuda')
   pending['images']=(batch['left_img'],batch['right_img']);sensors={k:batch[k] for k in SENSOR_INPUT_KEYS}
   arrays.update(left_input=batch['left_img'].cpu().numpy(),right_input=batch['right_img'].cpu().numpy(),
    image_shape=batch['image_shape'],cropped_P2=batch['calib'][0].P2,cropped_P3=batch['calib'][0].P3,crop_offsets=np.array(batch['calib'][0].offsets))
   with torch.no_grad():predictions,diagnostics=wrapped(sensors)
   torch.cuda.synchronize()
   assert calls==dict(detector=1,image_backbone=2,feature_neck=2,build_cost=1,head3D=1,depth_logits=1,forbidden=0),calls
   assert len(predictions)==1 and all(torch.isfinite(predictions[0][k]).all() for k in ('pred_boxes','pred_scores'))
   p=predictions[0];native=p['batch_dict']
   for k in ('pred_boxes','pred_scores','pred_labels'):arrays[k]=p[k].detach().cpu().numpy()
   for k in ('batch_box_preds','batch_cls_preds','depth_samples'):arrays[k]=native[k].detach().cpu().numpy()
   arrays['depth_pred']=native['depth_preds'][-1].detach().cpu().numpy();arrays['depth_pred_local']=native['depth_preds_local'][-1].detach().cpu().numpy()
   arrays['depth_axis_low']=b.downsampled_depth.detach().cpu().numpy();arrays.update(captured)
   assert all(np.isfinite(v).all() for v in arrays.values())
   after=state_record(model);assert after==initial
   target=OUT/(frame+'.npz');np.savez_compressed(target,**arrays)
   row=dict(frame_id=frame,artifact=str(target.relative_to(ROOT)),sha256=sha(target),bytes=target.stat().st_size,
    arrays={k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()) for k,v in arrays.items()},
    calls=dict(calls),inference_keys=list(SENSOR_INPUT_KEYS),prediction_count=len(arrays['pred_boxes']),readonly_states=after,
    diagnostic_keys=list(diagnostics),teacher_or_loss_calls=0)
   report['frames'].append(row);save();print(json.dumps(dict(frame=frame,count=row['prediction_count'])),flush=True)
   pending.clear();captured.clear();del predictions,diagnostics,native,p,sensors,batch,arrays,images
  report.update(state='passed_full484_native_sensor_only_PRO6000_forward_pending_actual_terminal_and_local_proof',
    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    final_states=state_record(model),DDP_backend=dist.get_backend())
 except BaseException:
  report['state']='failed';report['traceback']=traceback.format_exc();raise
 finally:
  for h in handles:h.remove()
  if dist.is_initialized():dist.destroy_process_group()
  if group is not None:group.cleanup()
  report['base_freeze_after']=freeze();report['base_runtime_unchanged']=report['base_freeze_after']==lock['base_freeze']
  report['sources_after']={p:sha(ROOT/p) for p in lock['frozen_inputs']}
  report['frozen_inputs_unchanged']=report['sources_after']==lock['frozen_inputs']
  report['ended_unix']=time.time();save()
  assert report['base_runtime_unchanged'] and report['frozen_inputs_unchanged']
if __name__=='__main__':main()
