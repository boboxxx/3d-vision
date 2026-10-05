"""Final-epoch, full validation inference; no evaluation labels in this process."""
import argparse, hashlib, importlib.util, json, os, subprocess, sys, tempfile, time, traceback, types
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def states(s):
 if hasattr(s,'state_dict'):s=s.state_dict()
 return {k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()) for k,v in s.items()}
def desc(a):return dict(dtype=str(a.dtype),shape=list(a.shape),sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
def freeze():
 e=os.environ.copy();e.pop('PYTHONPATH',None)
 return subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True,env=e)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=(17,),required=True);parser.add_argument('--arm',choices=('G','P','S','B'),required=True)
 parser.add_argument('--channel',choices=('identity','awgn','rayleigh'),required=True);parser.add_argument('--snr',type=int,choices=range(6,19),default=10)
 parser.add_argument('--native-training-audit',type=Path,required=True);parser.add_argument('--local-training-audit',type=Path,required=True)
 args=parser.parse_args();seed,arm=args.seed,args.arm
 assert args.channel!='identity' or args.snr==10
 assert args.channel!='rayleigh' or args.snr in (6,8,10,12,14,16,18)
 lockpath=ROOT/'experiments/cost-field/GPU-inputs-004.json';lock=json.loads(lockpath.read_text())
 training_dir=ROOT/'data/runs'/f'cost-field-full-seed{seed}-004';training_report=training_dir/'report.json';training=json.loads(training_report.read_text())
 assert training['seed']==seed and training['update_count']==44544 and len(training['completed_epochs'])==12
 assert training['state']=='finished_all44544_full_split_task_updates_pending_actual_terminal_and_complete_proof'
 assert training['input_sha256']==sha(lockpath)
 proofs=[];verified_training_ledgers=[]
 for proofpath in (args.native_training_audit,args.local_training_audit):
  proof=json.loads(proofpath.read_text())
  assert proof['state']=='passed_complete_training_stream' and proof['whole_training_closed'] and proof['seed']==seed and proof['checked_updates']==44544
  assert proof['report_sha256']==sha(training_report) and proof['updates_jsonl_sha256']==training['updates_jsonl_sha256']
  assert proof['actual_terminal']['raw_job_id']==training['job_id'] and proof['job_id']==training['job_id']
  assert proof['paired_physical_steps']==11136 and proof['paired_physical_comparisons']==33408
  assert proof['verifier_sha256']==sha(ROOT/'data/provenance/audit-cost-field-full-007.py')
  assert proof['protocol_sha256']==sha(ROOT/'experiments/cost-field/full-proof-protocol-007.md')
  verified_training_ledgers.append(proof['checked_record_ledger_sha256'])
  proofs.append(dict(path=str(proofpath),sha256=sha(proofpath)))
 assert len(set(verified_training_ledgers))==1
 assert args.native_training_audit.resolve()!=args.local_training_audit.resolve()
 sources={p:sha(ROOT/p) for p in lock['frozen_inputs']};assert sources==lock['frozen_inputs'] and freeze()==lock['base_freeze']
 own={str(p.relative_to(ROOT)):sha(p) for p in HERE.glob('*.py')}
 own['experiments/cost-field/final-validation-original-grid-008.md']=sha(HERE.parent/'final-validation-original-grid-008.md')
 own['experiments/cost-field/final-validation-protocol-005.md']=sha(HERE.parent/'final-validation-protocol-005.md')
 manifestpath=ROOT/lock['dataset_manifest_path'];assert sha(manifestpath)==lock['dataset_manifest_sha256']
 manifest=json.loads(manifestpath.read_text());ids=manifest['splits']['val']['ids'];assert len(ids)==len(set(ids))==3769
 assert not set(ids)&set(manifest['splits']['train']['ids'])
 epoch=[x for x in training['completed_epochs'] if x['arm']==arm and x['epoch']==3];assert len(epoch)==1
 final_checkpoint=ROOT/epoch[0]['checkpoint'];assert sha(final_checkpoint)==epoch[0]['sha256']
 out=ROOT/'data/runs'/f'cost-field-val-seed{seed}-{arm}-{args.channel}-{args.snr}-008';assert not out.exists();out.mkdir()
 predictions_dir=out/'data';predictions_dir.mkdir();wires_dir=out/'wires';wires_dir.mkdir()
 report=dict(state='starting',seed=seed,arm=arm,channel=args.channel,snr_db=args.snr,job_id=os.environ.get('SLURM_JOB_ID'),started_unix=time.time(),frames_complete=0,
  training_proofs=proofs,training_report_sha256=sha(training_report),final_codec_checkpoint=str(final_checkpoint.relative_to(ROOT)),final_codec_checkpoint_sha256=sha(final_checkpoint),
  validation_ids=ids,manifest_sha256=sha(manifestpath),sources_before=sources,inference_sources_before=own,base_freeze_before=lock['base_freeze'],AP_computed=False,
  labels_GT_LiDAR_at_inference=False,inference_noise_seed_rule='2027100000+integer(frame_id)',records_file=str((out/'frames.jsonl').relative_to(ROOT)))
 def save():(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 save();group=None;handles=[];records=None
 try:
  torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed)
  sys.path[:0]=[str(ROOT/'src'),str(ROOT/'third_party/LIGA-Stereo'),str(ROOT/'third_party/mmdetection_kitti'),str(ROOT/'experiments/original-detector-integration'),str(HERE.parent/'code_004'),str(HERE.parent)]
  builds=json.loads((ROOT/'data/engineering/artemis-detector-operators-build-001.json').read_text())
  for package,name in [('build_cost_volume','build_cost_volume_cuda'),('iou3d_nms','iou3d_nms_cuda'),('roiaware_pool3d','roiaware_pool3d_cuda')]:
   paths=[ROOT/p for p in builds['binaries'] if Path(p).name.startswith(name+'.')];assert len(paths)==1
   full='liga.ops.'+package+'.'+name;spec=importlib.util.spec_from_file_location(full,paths[0]);m=importlib.util.module_from_spec(spec);sys.modules[full]=m;spec.loader.exec_module(m)
  from easydict import EasyDict
  from liga.config import cfg_from_yaml_file
  from liga.datasets.stereo_dataset_template import StereoDatasetTemplate
  from liga.datasets.kitti.stereo_kitti_dataset import StereoKittiDataset
  from liga.models import build_network
  from liga.utils.calibration_kitti import Calibration
  from geocomm.compat import adapt_spconv_state
  from geocomm.inference import SENSOR_INPUT_KEYS
  from adapter import liga_preprocess
  from codec import CostFieldCodec,GenericDenseCodec,physical_channel
  from forward_with_cut import forward_with_cut
  import mmcv,torchvision
  assert torch.__version__=='2.7.1+cu128' and np.__version__=='1.26.4' and mmcv.__version__=='1.7.2' and torchvision.__version__=='0.22.1+cu128'
  assert 'artemis-framework-overlay-005/site-packages' in mmcv.__file__
  assert torch.cuda.device_count()==1 and torch.cuda.get_device_capability(0)==(12,0) and 'RTX PRO 6000' in torch.cuda.get_device_name(0)
  report['GPU']=torch.cuda.get_device_name(0)
  os.chdir(ROOT/'third_party/LIGA-Stereo')
  cfg=cfg_from_yaml_file(str(ROOT/'configs/diagnostic/clean_holdout.yaml'),EasyDict())
  cfg.MODEL.BACKBONE_3D.feature_backbone_pretrained=None;cfg.MODEL.LIDAR_MODEL.PRETRAINED_MODEL=None
  dataset=StereoDatasetTemplate(cfg.DATA_CONFIG,cfg.CLASS_NAMES,training=False,root_path=ROOT/'data/kitti')
  dataset.boxes_gt_in_cam2_view=cfg.DATA_CONFIG.BOXES_GT_IN_CAM2_VIEW
  assert cfg.CLASS_NAMES==['Car','Pedestrian','Cyclist']
  group=tempfile.TemporaryDirectory(prefix='val-liga-DDP-',dir=out)
  dist.init_process_group('nccl',init_method='file://'+group.name+'/rank',rank=0,world_size=1)
  model=build_network(cfg.MODEL,len(cfg.CLASS_NAMES),dataset).cuda().eval();b=model.backbone_3d
  assert all(x is None for x in (b.semantic_link,b.stereo_feature_link,b.student_semantic_link_encoder,model.rgb_semantic_link))
  checkpoint=ROOT/'assets/liga-native-001/liga-author-download'
  assert sha(checkpoint)=='3f36af30b686f23d0868482675177b29cc50c2b378577dd12f9092ddb0235b2e'
  raw=torch.load(checkpoint,map_location='cpu',weights_only=True)['model_state'];state=adapt_spconv_state(model,raw)
  assert len(state)==len(model.state_dict())==484 and set(state)==set(model.state_dict())
  model.load_state_dict(state,strict=True);initial=states(model);assert initial==training['loaded_states'];del raw,state
  report['author_initial_states']=initial
  axis=torch.linspace(b.CV_DEPTH_MIN,b.CV_DEPTH_MAX,len(b.downsampled_depth))
  codec=(GenericDenseCodec(axis,arm) if arm=='B' else CostFieldCodec(axis,arm)).cuda().eval()
  saved=torch.load(final_checkpoint,map_location='cpu',weights_only=True)['codec'];assert states(saved)==training['codec_final_states'][arm]
  codec.load_state_dict(saved,strict=True);codec_initial=states(codec);report['codec_initial_states']=codec_initial
  del saved
  for p in list(model.parameters())+list(codec.parameters()):p.requires_grad_(False)
  b.forward=types.MethodType(forward_with_cut,b)
  calls={};pending={};wire={}
  def forbidden(*unused,**kwargs):calls['forbidden']+=1;raise RuntimeError('teacher/GT/loss forbidden')
  handles.append(model.lidar_model.register_forward_pre_hook(forbidden))
  model.get_training_loss=forbidden;model.dense_head.get_loss=forbidden;model.depth_loss_head.get_loss=forbidden;model.dense_head_2d.get_loss=forbidden
  def entry(module,args0):
   assert not module.training and not torch.is_grad_enabled() and set(args0[0])==set(SENSOR_INPUT_KEYS)
   assert args0[0]['left_img'] is pending['images'][0] and args0[0]['right_img'] is pending['images'][1];calls['detector']+=1
  def backbone(module,args0):assert args0[0] is pending['images'][calls['image_backbone']];calls['image_backbone']+=1
  def count(key):
   def hook(module,args0):calls[key]+=1
   return hook
  handles.extend([model.register_forward_pre_hook(entry),b.feature_backbone.register_forward_pre_hook(backbone),b.feature_neck.register_forward_pre_hook(count('feature_neck')),b.build_cost.register_forward_pre_hook(count('build_cost')),model.dense_head.register_forward_pre_hook(count('head3D')),b.pred_stereo[-1].register_forward_hook(lambda *unused: calls.__setitem__('depth_logits',calls['depth_logits']+1))])
  def communication(cost,appearance,batch):
   calls['cut']+=1;prior=b.pred_stereo[-1][:3](cost).softmax(2)
   tx,private=codec.encode(cost,appearance,prior)
   generator=torch.Generator(device='cuda').manual_seed(2027100000+int(pending['frame']))
   rx,physical=physical_channel(tx,args.channel,args.snr,generator)
   received_cost,received_app=codec.decode(rx,cost.shape[-2:],appearance.shape[-2:]);private.clear()
   again_cost,again_app=codec.decode(rx,cost.shape[-2:],appearance.shape[-2:]);assert torch.equal(received_cost,again_cost) and torch.equal(received_app,again_app)
   wire.update(wire_tx=tx.cpu().numpy().copy(),wire_received=rx.cpu().numpy().copy(),**{'channel_'+k:physical[k].cpu().numpy().copy() for k in ('noise','fading','received_baseband')})
   pending.update(decoded_cost=desc(received_cost.cpu().numpy()),decoded_appearance=desc(received_app.cpu().numpy()),cost_hw=list(cost.shape[-2:]),appearance_hw=list(appearance.shape[-2:]))
   assert tx.shape[1]==(cost.shape[-2]//4)*(cost.shape[-1]//4)*19
   return received_cost,received_app
  b._communication_callback=communication;records=(out/'frames.jsonl').open('x',buffering=1)
  torch.cuda.reset_peak_memory_stats();report['state']='running_full3769_final_epoch_inference';save()
  with torch.no_grad():
   for index,frame in enumerate(ids):
    pending.clear();pending['frame']=frame;wire.clear();calls.update(detector=0,image_backbone=0,feature_neck=0,build_cost=0,head3D=0,depth_logits=0,forbidden=0,cut=0)
    images=[];source={}
    for folder in ('image_2','image_3'):
     path=ROOT/f'data/kitti/training/{folder}/{frame}.png';item=manifest['files'][str(path.relative_to(ROOT))]
     assert sha(path)==item['sha256'] and path.stat().st_size==item['bytes'];source[folder]=item
     with Image.open(path) as image:rgb=np.asarray(image.convert('RGB')).copy()
     images.append(torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).float()/255.)
    path=ROOT/f'data/kitti/training/calib/{frame}.txt';item=manifest['files'][str(path.relative_to(ROOT))];assert sha(path)==item['sha256'];source['calib']=item
    cal=Calibration(str(path));batch=liga_preprocess(images,cal,frame,dataset.data_augmentor,dataset.collate_batch,device='cuda')
    pending['images']=(batch['left_img'],batch['right_img']);sensor={k:batch[k] for k in SENSOR_INPUT_KEYS}
    arrays=dict(original_P2=cal.P2.copy(),original_P3=cal.P3.copy(),image_shape=batch['image_shape'],cropped_P2=batch['calib'][0].P2.copy(),cropped_P3=batch['calib'][0].P3.copy(),crop_offsets=np.array(batch['calib'][0].offsets))
    predictions,diagnostics=model(sensor);torch.cuda.synchronize()
    assert calls==dict(detector=1,image_backbone=2,feature_neck=2,build_cost=1,head3D=1,depth_logits=1,forbidden=0,cut=1)
    assert len(predictions)==1
    for key in ('pred_boxes','pred_scores','pred_labels'):arrays[key]=predictions[0][key].cpu().numpy().copy()
    StereoKittiDataset.generate_prediction_dicts(dataset,batch,predictions,cfg.CLASS_NAMES,output_path=predictions_dir)
    text=predictions_dir/(frame+'.txt');lines=text.read_text().splitlines();assert len(lines)==len(arrays['pred_boxes'])
    assert all(len(line.split())==16 and line.split()[0] in cfg.CLASS_NAMES and np.isfinite([float(v) for v in line.split()[1:]]).all() for line in lines)
    arrays.update(wire);assert all(np.isfinite(v).all() for v in arrays.values())
    tx=arrays['wire_tx'];uses=tx.shape[1];energy=float(np.square(tx.astype(np.float64)).sum());assert abs(energy-uses)<=uses*1e-6
    assert states(model)==initial and states(codec)==codec_initial and all(p.grad is None for p in list(model.parameters())+list(codec.parameters()))
    block=wires_dir/f'block{index//256:04d}';block.mkdir(exist_ok=True);artifact=block/(frame+'.npz');np.savez_compressed(artifact,**arrays)
    row=dict(index=index,frame_id=frame,arm=arm,seed=seed,channel=args.channel,snr_db=args.snr,source=source,prediction=str(text.relative_to(ROOT)),prediction_sha256=sha(text),prediction_count=len(lines),
     artifact=str(artifact.relative_to(ROOT)),sha256=sha(artifact),bytes=artifact.stat().st_size,arrays={k:desc(v) for k,v in arrays.items()},calls=dict(calls),inference_keys=list(SENSOR_INPUT_KEYS),
     decoded_cost=pending['decoded_cost'],decoded_appearance=pending['decoded_appearance'],cost_hw=pending['cost_hw'],appearance_hw=pending['appearance_hw'],complex_uses=uses,energy=energy,
     received_only_decode_identical_after_private_clear=True,original_states_sha256=hashlib.sha256(json.dumps(initial,sort_keys=True).encode()).hexdigest(),codec_states_sha256=hashlib.sha256(json.dumps(codec_initial,sort_keys=True).encode()).hexdigest())
    records.write(json.dumps(row,allow_nan=False)+'\n');report['frames_complete']=index+1
    if (index+1)%32==0:save();print(json.dumps(dict(frames=index+1,seed=seed,arm=arm,channel=args.channel,snr_db=args.snr)),flush=True)
    del images,batch,sensor,predictions,diagnostics,arrays;pending.clear();wire.clear();model.dense_head.forward_ret_dict.clear()
  records.close();records=None
  assert {p.stem for p in predictions_dir.glob('*.txt')}==set(ids)
  report.update(state='finished_full3769_final_epoch_inference_pending_terminal_wire_text_AP_closure',frames_jsonl_sha256=sha(out/'frames.jsonl'),author_final_states=states(model),codec_final_states=states(codec),peak_reserved_bytes=torch.cuda.max_memory_reserved(),base_freeze_after=freeze(),sources_after={p:sha(ROOT/p) for p in lock['frozen_inputs']},inference_sources_after={p:sha(ROOT/p) for p in own})
  assert report['sources_after']==sources and report['inference_sources_after']==own and report['base_freeze_after']==lock['base_freeze']
  assert sha(final_checkpoint)==report['final_codec_checkpoint_sha256'] and sha(training_report)==report['training_report_sha256']
 except BaseException:report['state']='failed';report['traceback']=traceback.format_exc();raise
 finally:
  if records:records.close()
  for handle in handles:handle.remove()
  if dist.is_initialized():dist.destroy_process_group()
  if group:group.cleanup()
  report['ended_unix']=time.time();save()
if __name__=='__main__':main()
