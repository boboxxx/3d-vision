"""Full native received-only conditional cost-field actual detection-loss gate."""
import argparse
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
INPUT=HERE.parent/'GPU-inputs-004.json'
parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=(17,23,41),required=True)
ARGS=parser.parse_args()
OUT=ROOT/'data/runs'/('cost-field-full-seed'+str(ARGS.seed)+'-004')

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
 report=dict(seed=ARGS.seed,update_count=0,completed_epochs=[],state='starting',stage='check_inputs',started_unix=time.time(),job_id=os.environ.get('SLURM_JOB_ID'),
   input_sha256=sha(INPUT),records_file=str((OUT/"updates.jsonl").relative_to(ROOT)),labels_loss_side_only=True,GT_or_labels_or_LiDAR_at_forward_entry=False,optimizer_used=True)
 def save():(OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 save();group=None;handles=[]
 try:
  torch.set_num_threads(2);torch.manual_seed(ARGS.seed);np.random.seed(ARGS.seed)
  sources={p:sha(ROOT/p) for p in lock['frozen_inputs']};assert sources==lock['frozen_inputs']
  report['sources_before']=sources;report['base_freeze_before']=freeze();assert report['base_freeze_before']==lock['base_freeze']
  for p,item in lock['sensor_inputs'].items():assert sha(ROOT/p)==item['sha256'] and (ROOT/p).stat().st_size==item['bytes']
  dataset_manifest_path=ROOT/lock['dataset_manifest_path']
  assert sha(dataset_manifest_path)==lock['dataset_manifest_sha256']
  manifest=json.loads(dataset_manifest_path.read_text())
  assert manifest['state']=='passed_whole_raw_KITTI_stereo_train3712_val3769_bytes_and_disjoint_splits'
  train_ids=manifest['splits']['train']['ids'];val_ids=manifest['splits']['val']['ids']
  assert len(train_ids)==3712 and len(val_ids)==3769 and not set(train_ids)&set(val_ids)
  def check_dataset():
   for rel,item in manifest['files'].items():assert (ROOT/rel).stat().st_size==item['bytes'] and sha(ROOT/rel)==item['sha256'],rel
   for item in manifest['splits'].values():assert sha(ROOT/item['path'])==item['sha256']
  check_dataset();report['dataset_manifest_sha256']=sha(dataset_manifest_path);save()
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
  from liga.utils import box_utils, object3d_kitti
  from geocomm.compat import adapt_spconv_state
  from geocomm.inference import SENSOR_INPUT_KEYS
  from adapter import liga_preprocess
  from codec import CostFieldCodec, GenericDenseCodec, physical_channel
  sys.path.append(str(HERE.parent))
  sys.path.append(str(HERE.parent))
  from forward_with_cut import forward_with_cut
  import types
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
  initial=report['loaded_states']
  axis=torch.linspace(b.CV_DEPTH_MIN,b.CV_DEPTH_MAX,len(b.downsampled_depth),device='cpu')
  report['depth_coordinate_contract']=dict(voxel_query_axis=axis.tolist(),source_cost_sweep=b.downsampled_depth.cpu().tolist(),depth_head_centers=b.depth.cpu().tolist(),native_sampler_align_corners=True)
  codecs={arm:CostFieldCodec(axis,arm).cuda() for arm in ('G','P','S')}
  common={k:v.detach().clone() for k,v in codecs['G'].state_dict().items()}
  for codec in codecs.values():codec.load_state_dict(common)
  torch.manual_seed(ARGS.seed)
  codecs['B']=GenericDenseCodec(axis,'B').cuda()
  for name in ('appearance_encoder','appearance_decoder'):
   getattr(codecs['B'],name).load_state_dict(getattr(codecs['P'],name).state_dict())
  optimizers={arm:torch.optim.Adam(codec.parameters(),lr=1e-3,weight_decay=0) for arm,codec in codecs.items()}
  report['codec_initial_states']={arm:state_record(codec) for arm,codec in codecs.items()}
  report['codec_parameters_per_arm']={arm:sum(p.numel() for p in codec.parameters()) for arm,codec in codecs.items()}
  assert report['codec_parameters_per_arm']==dict(G=1650,P=1650,S=1650,B=1751)
  torch.save({arm:{k:v.cpu() for k,v in codec.state_dict().items()} for arm,codec in codecs.items()},OUT/'codec-initial.pt')
  report['codec_initial_checkpoint_sha256']=sha(OUT/'codec-initial.pt')
  b.forward=types.MethodType(forward_with_cut,b)
  wire={};current={}
  def communication(cost,appearance,batch):
   current['cut_calls']+=1
   # Identical original source head is computed for all three arms.
   prior=b.pred_stereo[-1][:3](cost).softmax(2)
   codec=codecs[current['arm']]
   tx,private=codec.encode(cost,appearance,prior)
   generator=torch.Generator(device='cuda').manual_seed(ARGS.seed*1000000+current['step'])
   rx,physical=physical_channel(tx,current['channel'],current['snr_db'],generator)
   received_cost,received_appearance=codec.decode(rx,cost.shape[-2:],appearance.shape[-2:])
   # Delete the explicit sender-private state before repeating the receiver.
   private.clear();again_cost,again_appearance=codec.decode(rx,cost.shape[-2:],appearance.shape[-2:])
   assert torch.equal(received_cost,again_cost) and torch.equal(received_appearance,again_appearance)
   wire.update(tx=tx.detach().cpu().numpy().copy(),received=rx.detach().cpu().numpy().copy(),
    energy=float(tx.detach().double().square().sum()),complex_uses=tx.shape[1],physical_arrays={k:physical[k].detach().cpu().numpy().copy() for k in ('noise','fading','received_baseband')},
    decoded_cost=dict(shape=list(received_cost.shape),dtype=str(received_cost.dtype),sha256=hashlib.sha256(received_cost.detach().cpu().contiguous().numpy().tobytes()).hexdigest()),
    decoded_appearance=dict(shape=list(received_appearance.shape),dtype=str(received_appearance.dtype),sha256=hashlib.sha256(received_appearance.detach().cpu().contiguous().numpy().tobytes()).hexdigest()))
   assert tx.shape[1]==(cost.shape[-2]//4)*(cost.shape[-1]//4)*19
   assert abs(wire['energy']-wire['complex_uses'])<=wire['complex_uses']*1e-6
   return received_cost,received_appearance
  b._communication_callback=communication
  for param in model.parameters():param.requires_grad_(False)
  assert all(not param.requires_grad for param in model.parameters())
  calls={};pending={};captured={}
  def forbidden(*args,**kwargs):calls['forbidden']+=1;raise RuntimeError('teacher or training loss forbidden')
  handles.append(model.lidar_model.register_forward_pre_hook(forbidden))
  model.get_training_loss=forbidden;model.dense_head.get_loss=forbidden;model.depth_loss_head.get_loss=forbidden
  model.dense_head_2d.get_loss=forbidden
  def entry(module,args):
   assert not module.training and torch.is_grad_enabled() and set(args[0])==set(SENSOR_INPUT_KEYS)
   assert args[0]['left_img'] is pending['images'][0] and args[0]['right_img'] is pending['images'][1]
   calls['detector']+=1
  def backbone(module,args):
   assert args[0] is pending['images'][calls['image_backbone']];calls['image_backbone']+=1
  def count(key):
   def hook(module,args):calls[key]+=1
   return hook
  def logit(module,args,output):
   assert isinstance(output,torch.Tensor) and torch.isfinite(output).all()
   calls['depth_logits']+=1
  handles.extend([model.register_forward_pre_hook(entry),b.feature_backbone.register_forward_pre_hook(backbone),
   b.feature_neck.register_forward_pre_hook(count('feature_neck')),b.build_cost.register_forward_pre_hook(count('build_cost')),
   model.dense_head.register_forward_pre_hook(count('head3D')),b.pred_stereo[-1].register_forward_hook(logit)])
  torch.cuda.reset_peak_memory_stats();report['stage']='actual_sensor_only_forward_then_supervised_loss_backward';report['state']='running_full_split_task_training';save()
  rng=np.random.default_rng(ARGS.seed+200000)
  schedule=[('awgn' if rng.integers(0,2)==0 else 'rayleigh',float(rng.uniform(6,18))) for _ in range(3*3712)]
  order=[np.random.default_rng(ARGS.seed+1000*epoch).permutation(train_ids).tolist() for epoch in range(3)]
  report['training_schedule']=dict(per_epoch_order=order,channels_and_snr=schedule)
  save()
  records=(OUT/'updates.jsonl').open('x',buffering=1)
  cases=[(frame,arm,*schedule[epoch*3712+index],epoch*3712+index,epoch,index) for arm in ('G','P','S','B') for epoch in range(3) for index,frame in enumerate(order[epoch])]
  for frame,arm,channel,snr_db,step,epoch,index in cases:
   update_dir=OUT/'updates'/arm/('epoch'+str(epoch+1))/('block'+str(step//256).zfill(4))
   update_dir.mkdir(parents=True,exist_ok=True)
   current.update(frame=frame,arm=arm,channel=channel,snr_db=snr_db,step=step,cut_calls=0);wire.clear()
   for codec in codecs.values():codec.zero_grad(set_to_none=True)
   codec_before=state_record(codecs[arm])
   calls.update(detector=0,image_backbone=0,feature_neck=0,build_cost=0,head3D=0,depth_logits=0,forbidden=0)
   arrays={};images=[]
   for view,folder in [('left','image_2'),('right','image_3')]:
    p=ROOT/'data/kitti/training'/folder/(frame+'.png');assert sha(p)==manifest['files'][str(p.relative_to(ROOT))]['sha256']
    with Image.open(p) as im:rgb=np.asarray(im.convert('RGB')).copy()
    images.append(torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).float()/255.)
   p=ROOT/'data/kitti/training/calib'/(frame+'.txt');assert sha(p)==manifest['files'][str(p.relative_to(ROOT))]['sha256']
   cal=Calibration(str(p));arrays.update(original_P2=cal.P2.copy(),original_P3=cal.P3.copy())
   batch=liga_preprocess(images,cal,frame,dataset.data_augmentor, dataset.collate_batch,device='cuda')
   batch['left_img']=batch['left_img'].detach()
   batch['right_img']=batch['right_img'].detach()
   pending['images']=(batch['left_img'],batch['right_img']);sensors={k:batch[k] for k in SENSOR_INPUT_KEYS}
   arrays.update(left_input=batch['left_img'].detach().cpu().numpy(),right_input=batch['right_img'].detach().cpu().numpy(),
    image_shape=batch['image_shape'],cropped_P2=batch['calib'][0].P2,cropped_P3=batch['calib'][0].P3,crop_offsets=np.array(batch['calib'][0].offsets))
   predictions,diagnostics=model(sensors)
   torch.cuda.synchronize()
   assert calls==dict(detector=1,image_backbone=2,feature_neck=2,build_cost=1,head3D=1,depth_logits=1,forbidden=0),calls
   assert len(predictions)==1 and all(torch.isfinite(predictions[0][k]).all() for k in ('pred_boxes','pred_scores'))
   p=predictions[0];native=p['batch_dict']
   for k in ('pred_boxes','pred_scores','pred_labels'):arrays[k]=p[k].detach().cpu().numpy()
   # Supervision is read only after the sensor-only model forward has returned.
   label=ROOT/'data/kitti/training/label_2'/(frame+'.txt')
   assert sha(label)==manifest['files'][str(label.relative_to(ROOT))]['sha256']
   objects=object3d_kitti.get_objects_from_label(label)
   camera=[];classes=[]
   for obj in objects:
    cls={'Van':'Car','Person_sitting':'Pedestrian'}.get(obj.cls_type,obj.cls_type)
    if cls not in cfg.CLASS_NAMES:continue
    camera.append([*obj.loc,obj.l,obj.h,obj.w,obj.ry]);classes.append(cfg.CLASS_NAMES.index(cls)+1)
   camera=np.asarray(camera,np.float32).reshape(-1,7)
   lidar=box_utils.boxes3d_kitti_camera_to_lidar(camera,cal,pseudo_lidar=True,pseudo_cam2_view=False)
   mask=box_utils.mask_boxes_outside_range_numpy(lidar,np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE),min_num_corners=1)
   gt=np.concatenate([lidar[mask],np.asarray(classes,np.float32)[mask,None]],axis=1)
   targets=model.dense_head.assign_targets(torch.from_numpy(gt[None]).cuda())
   positive=int((targets['box_cls_labels']>0).sum())
   model.dense_head.forward_ret_dict.update(targets)
   cls_loss,cls_stats=model.dense_head.get_cls_layer_loss()
   box_loss,box_stats=model.dense_head.get_box_reg_layer_loss()
   loss=cls_loss+box_loss;assert torch.isfinite(loss)
   report['stage']='actual_native_detection_loss_backward';save()
   assert current['cut_calls']==1
   assert loss.requires_grad;loss.backward();torch.cuda.synchronize()
   codec=codecs[arm]
   for name,param in codec.named_parameters():
    # Save before gate assertions, so a failed gradient is retained as data.
    arrays['codec_gradient_'+name]=None if param.grad is None else param.grad.detach().cpu().numpy().copy()
   if any(v is None or not np.isfinite(v).all() for k,v in arrays.items() if k.startswith('codec_gradient_')):
    np.savez_compressed(OUT/(frame+'-'+arm+'-step'+str(step)+'-failed-gradient.npz'),**{k:v for k,v in arrays.items() if v is not None})
    raise RuntimeError('Nonfinite or absent actual native task gradient; saved failure')
   gradient_norm=torch.nn.utils.clip_grad_norm_(codec.parameters(),10,error_if_nonfinite=True)
   for name,param in codec.named_parameters():arrays['clipped_gradient_'+name]=param.grad.detach().cpu().numpy().copy()
   optimizers[arm].step()
   codec_after=state_record(codec)
   changed=[k for k in codec_before if codec_before[k]!=codec_after[k]]
   assert all(torch.isfinite(p).all() for p in codec.parameters())
   checkpoint_path=update_dir/(frame+'-step'+str(step)+'-state.pt')
   def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value
   torch.save(cpu_tree(dict(codec=codec.state_dict(),optimizer=optimizers[arm].state_dict())),checkpoint_path)
   arrays.update(wire_tx=wire['tx'],wire_received=wire['received'],**{'channel_'+k:v for k,v in wire['physical_arrays'].items()})
   assert all(param.grad is None for param in model.parameters())
   arrays.pop('left_input');arrays.pop('right_input')
   arrays.update(camera_gt=camera,range_keep_mask=mask,gt_boxes=gt)
   for key,value in targets.items():
    assert isinstance(value,torch.Tensor);arrays['assigned_'+key]=value.detach().cpu().numpy().copy()
   arrays['losses']=np.array([cls_loss.item(),box_loss.item(),loss.item()],np.float64)
   assert all(np.isfinite(v).all() for v in arrays.values())
   after=state_record(model);assert after==initial
   target=update_dir/(frame+'-step'+str(step)+'.npz');np.savez_compressed(target,**arrays)
   row=dict(frame_id=frame,arm=arm,epoch=epoch+1,epoch_index=index,channel=channel,snr_db=snr_db,step=step,cut_calls=current['cut_calls'],artifact=str(target.relative_to(ROOT)),sha256=sha(target),bytes=target.stat().st_size,
    arrays={k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()) for k,v in arrays.items()},
    calls=dict(calls),inference_keys=list(SENSOR_INPUT_KEYS),prediction_count=len(arrays['pred_boxes']),readonly_states_sha256=hashlib.sha256(json.dumps(after,sort_keys=True).encode()).hexdigest(),
    diagnostic_keys=list(diagnostics),forbidden_calls=0,positive_anchors=positive,
    cls_stats=cls_stats,box_stats=box_stats,all_parameter_gradients_absent=True,
    label_file=str(label.relative_to(ROOT)),label_sha256=sha(label),
    complex_uses=wire.get('complex_uses',0),energy=wire.get('energy',0),
    received_only_decode_identical_after_private_clear=True,decoded_cost=wire.get('decoded_cost'),decoded_appearance=wire.get('decoded_appearance'),
    codec_states_before=codec_before,codec_states_after=codec_after,changed_states=changed,gradient_norm_before_clip=float(gradient_norm),optimizer_checkpoint=str(checkpoint_path.relative_to(ROOT)),optimizer_checkpoint_sha256=sha(checkpoint_path),
    codec_gradient_nonzero={key:int(np.count_nonzero(value)) for key,value in arrays.items() if key.startswith('codec_gradient_')})
   records.write(json.dumps(row,allow_nan=False)+'\n')
   report['update_count']+=1
   report['last_update']=dict(arm=arm,epoch=epoch+1,index=index,step=step,frame=frame,loss=float(loss.detach()),complex_uses=row['complex_uses'])
   if report['update_count']%32==0:
    save();print(json.dumps(report['last_update']),flush=True)
   if index==3711:
    assert state_record(model)==initial
    check_dataset()
    epoch_path=OUT/('checkpoint-'+arm+'-epoch'+str(epoch+1)+'.pt');shutil.copyfile(checkpoint_path,epoch_path)
    report['completed_epochs'].append(dict(arm=arm,epoch=epoch+1,updates=3712,checkpoint=str(epoch_path.relative_to(ROOT)),sha256=sha(epoch_path)))
    assert {p:sha(ROOT/p) for p in lock['frozen_inputs']}==lock['frozen_inputs']
    save()

   pending.clear();captured.clear();del predictions,diagnostics,native,p,sensors,batch,arrays,images,targets,loss,cls_loss,box_loss
   wire.clear()
   model.dense_head.forward_ret_dict.clear()
  records.close();assert report['update_count']==44544 and len(report['completed_epochs'])==12
  check_dataset()
  report['updates_jsonl_sha256']=sha(OUT/'updates.jsonl')
  report.update(state='finished_all44544_full_split_task_updates_pending_actual_terminal_and_complete_proof',
    codec_final_states={arm:state_record(codec) for arm,codec in codecs.items()},peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    final_states=state_record(model),distributed_backend=dist.get_backend(),all_parameters_frozen=True,all_parameter_gradients_absent=all(param.grad is None for param in model.parameters()))
 except BaseException:
  report['state']='failed';report['traceback']=traceback.format_exc();raise
 finally:
  for h in handles:h.remove()
  if dist.is_initialized():dist.destroy_process_group()
  if group is not None:group.cleanup()
  report['base_freeze_after']=freeze();report['base_runtime_unchanged']=report['base_freeze_after']==lock['base_freeze']
  report['sources_after']={p:sha(ROOT/p) for p in lock['frozen_inputs']}
  report['sensor_inputs_after']={p:dict(sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size) for p in lock['sensor_inputs']}
  assert report['sensor_inputs_after']==lock['sensor_inputs']
  report['frozen_inputs_unchanged']=report['sources_after']==lock['frozen_inputs']
  report['ended_unix']=time.time();save()
  if report['state']!='failed':assert all(report['codec_initial_states'][arm]!=report['codec_final_states'][arm] for arm in codecs)
  assert report['base_runtime_unchanged'] and report['frozen_inputs_unchanged']
if __name__=='__main__':main()
