"""Three native author LIGA GPU RGB endpoints, engineering only and no AP."""
import argparse,importlib.util,json,os,sys,tempfile,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];LIGA=ROOT/'third_party/LIGA-Stereo'
sys.path[:0]=[str(HERE),str(ROOT/'src'),str(LIGA),str(ROOT/'third_party/mmdetection_kitti'),str(ROOT/'reproduction/cao2025')]
from adapter import liga_preprocess,transmit_rgb
from data import StereoRGB
from wireless import WirelessVariant
import radio
from geocomm.evidence import sha256,source_identity
from geocomm.pooling_diagnostic import state_hashes
from geocomm.inference import SENSOR_INPUT_KEYS
from geocomm.compat import adapt_spconv_state
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output=args.output.resolve()
    if args.output.exists():p.error('preserve previous engineering; unique output')
    closure=json.loads((ROOT/'data/provenance/stereo-native-task-seed17-002-closure.json').read_text())
    if closure['state']!='closed_all_audits_passed':raise RuntimeError('F6b terminal fullsourceclosure required')
    usage=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    if len(usage)!=1:raise RuntimeError('singleGPU required')
    used,total=(int(v.strip()) for v in usage[0].split(','));free=(total-used)*2**20
    if free<12*2**30:raise RuntimeError('need12GiB NVIDIAphysicalfree without disturbing originaltraining')
    specs={'project':(ROOT,['src','scripts','configs','pyproject.toml']),
     'liga':(LIGA,['liga','configs','tools','setup.py']),'mmdet':(ROOT/'third_party/mmdetection_kitti',['mmdet']),
     'stereo_rcnn':(ROOT/'third_party/Stereo-RCNN',['lib','demo.py','test_net.py'])}
    sources={name:source_identity(*spec) for name,spec in specs.items()}
    original=json.loads((ROOT/'data/provenance/cao2025-staged-trainer-source-bfee9d8.json').read_text())
    if len(original)!=21 or any(sha256(ROOT/'reproduction/cao2025'/name)!=v for name,v in original.items()):raise RuntimeError('original sources changed')
    info=dict(state='starting',scope='native author LIGA threeRGB engineering endpoints, no AP',started_at_unix=time.time(),
     source_identities=sources,original21_sources=original,NVIDIA_free_before_bytes=free,conditions=[],device='cuda',codec_device='cpu')
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(info,indent=2,allow_nan=False)+'\n')
    save();group=None;handles=[]
    try:
        torch.set_num_threads(2);torch.manual_seed(17);np.random.seed(17);torch.cuda.manual_seed_all(17);os.chdir(LIGA)
        from easydict import EasyDict
        from liga.config import cfg_from_yaml_file
        from liga.datasets import build_dataloader
        from liga.datasets.augmentor.stereo_data_augmentor import StereoDataAugmentor
        from liga.datasets.stereo_dataset_template import StereoDatasetTemplate
        from liga.models import build_network
        from liga.utils.calibration_kitti import Calibration
        from liga.utils.common_utils import create_logger
        cfgpath=ROOT/'configs/diagnostic/clean_holdout.yaml';cfg=cfg_from_yaml_file(str(cfgpath),EasyDict())
        dataset,_,_=build_dataloader(cfg.DATA_CONFIG,cfg.CLASS_NAMES,batch_size=1,dist=False,workers=0,training=False,logger=create_logger())
        cache=StereoRGB('/mnt/d/paper6/data/kitti',ROOT/'data/engineering/cao2025-roi-holdout-001.jsonl',
         ROOT/'data/engineering/cao2025-roi-holdout-audit-001.json',ROOT/'data/internal-tuning-fold-001.json','geocomm_tune_holdout')
        frame=cache[0]
        if frame['frame_id']!='000036':raise RuntimeError('fixedfirstframe required')
        calibration=Calibration('/mnt/d/paper6/data/kitti/training/calib/000036.txt')
        augmentor=StereoDataAugmentor(dataset.root_path,cfg.DATA_CONFIG.TEST_DATA_AUGMENTOR,cfg.CLASS_NAMES)
        group=tempfile.TemporaryDirectory(prefix='original-LIGA-RGB-GPU-')
        dist.init_process_group('nccl',init_method='file://'+group.name+'/rank',rank=0,world_size=1)
        model=build_network(cfg.MODEL,len(cfg.CLASS_NAMES),dataset).cuda().eval()
        b=model.backbone_3d
        if any(v is not None for v in (b.semantic_link,b.stereo_feature_link,b.student_semantic_link_encoder,model.rgb_semantic_link)):
            raise RuntimeError('author484-state architecture must disable all new links/student')
        checkpoint=Path('/mnt/d/paper6/checkpoints/liga-author-download');author_sha='3f36af30b686f23d0868482675177b29cc50c2b378577dd12f9092ddb0235b2e'
        if sha256(checkpoint)!=author_sha:raise RuntimeError('verified author checkpoint differs')
        raw=torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state'];state=adapt_spconv_state(model,raw)
        actual=model.state_dict()
        if len(actual)!=484 or set(state)!=set(actual) or any(v.shape!=actual[k].shape or v.dtype!=actual[k].dtype or not torch.isfinite(v).all() for k,v in state.items()):
            raise RuntimeError('complete484-state native load differs')
        model.load_state_dict(state,strict=True)
        if any(not torch.equal(v.detach().cpu(),state[k]) for k,v in model.state_dict().items()):raise RuntimeError('loaded values differ')
        before=state_hashes(model);del raw,state,actual
        wrapped=torch.nn.parallel.DistributedDataParallel(model,device_ids=[0],broadcast_buffers=False)
        calls={};pending={}
        def forbidden(*a):calls['forbidden']+=1;raise RuntimeError('privileged teacher/trainingloss called')
        if model.lidar_model is not None:handles.append(model.lidar_model.register_forward_pre_hook(forbidden))
        model.get_training_loss=forbidden;model.dense_head.get_loss=forbidden;model.depth_loss_head.get_loss=forbidden
        if model.dense_head_2d is not None:model.dense_head_2d.get_loss=forbidden
        def input_scope(module,args):
            if module.training or torch.is_grad_enabled() or set(args[0])!=set(SENSOR_INPUT_KEYS):raise RuntimeError('non-sensor/nativeeval model entry')
            pending['images']=(args[0]['left_img'],args[0]['right_img'])
        def backbone_inputs(module,args):
            calls['image_backbone']+=1
            if args[0] is not pending['images'][calls['image_backbone']-1]:raise RuntimeError('clean image bypass')
        def count(name):
            def hook(module,args):calls[name]+=1
            return hook
        handles.extend([model.register_forward_pre_hook(input_scope),b.feature_backbone.register_forward_pre_hook(backbone_inputs),
          b.feature_neck.register_forward_pre_hook(count('feature_neck')),b.build_cost.register_forward_pre_hook(count('build_cost')),
          model.dense_head.register_forward_pre_hook(count('head3D'))])
        info.update(required_detector_states=484,detector_checkpoint_sha256=author_sha,detector_initial_state_hashes=before,config_sha256=sha256(cfgpath))
        spec=importlib.util.spec_from_file_location('paper6_cao2025_semantic',ROOT/'reproduction/cao2025/model.py')
        semantic=importlib.util.module_from_spec(spec);spec.loader.exec_module(semantic)
        codec=WirelessVariant(semantic.SemanticVariant('/mnt/d/paper6/checkpoints/spynet_sintel_final-3d2a1287.pth')).cpu().eval()
        initializer=Path('/mnt/d/paper6/runs/cao2025-full-native-seed17-001-stage1/initialization.pth')
        if sha256(initializer)!='f4d0d5d5bcc203ee5fd0554a0e691b2a530c56cb55d48fe21b6fc0f97331d3cb':raise RuntimeError('freshcodec initializer differs')
        initial=torch.load(initializer,map_location='cpu',weights_only=False)['model_state']
        if len(initial)!=768 or len(codec.state_dict())!=768:raise RuntimeError('fulloriginal768 layout differs')
        codec.load_state_dict(initial,strict=True);del initial;codec_before=state_hashes(codec)
        codec.forward=forbidden;codec.semantic_mse=forbidden
        info.update(codec_initialization_sha256=sha256(initializer),required_codec_states=768,codec_initial_state_hashes=codec_before,frame_id='000036')
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for condition in ('clean_relay','identity','awgn'):
                calls.update(image_backbone=0,feature_neck=0,build_cost=0,head3D=0,forbidden=0);pending.clear()
                if condition=='clean_relay':result=dict(outputs=(frame['left'],frame['right']),accounting=None,erasure=None,decoder_range=None)
                else:result=transmit_rgb(codec,radio,frame['left'],frame['right'],frame['boxes'],condition,10.,torch.Generator().manual_seed(17))
                row=dict(condition=condition,erasure=result['erasure'],accounting=result['accounting'],decoder_range=result['decoder_range'])
                if result['outputs'] is not None:
                    batch=liga_preprocess(result['outputs'],calibration,'000036',augmentor,StereoDatasetTemplate.collate_batch,device='cuda')
                    sensors={k:batch[k] for k in SENSOR_INPUT_KEYS};predictions,diagnostics=wrapped(sensors);torch.cuda.synchronize()
                    if calls!=dict(image_backbone=2,feature_neck=2,build_cost=1,head3D=1,forbidden=0):raise RuntimeError('native imagebackbone/receiver count differs')
                    if any(not torch.isfinite(p[k]).all() for p in predictions for k in ('pred_boxes','pred_scores')):raise RuntimeError('nonfinite native3D predictions')
                    row.update(prediction_count=len(predictions[0]['pred_boxes']),processed_image_shape=list(batch['left_img'].shape),
                        original_image_shape=batch['image_shape'].tolist(),crop_offsets=batch['calib'][0].offsets,inference_input_keys=sorted(sensors))
                    del predictions,diagnostics,batch,sensors;pending.clear()
                else:row.update(prediction_count=0,inference_input_keys=[])
                row['calls']=dict(calls)
                if state_hashes(model)!=before or state_hashes(codec)!=codec_before:raise RuntimeError('model/codec states mutated')
                if torch.cuda.max_memory_reserved()+2*2**30>free:raise RuntimeError('measured GPUendpoint lacks2GiB physicalmargin')
                info['conditions'].append(row);save();print(json.dumps({'condition':condition,'prediction_count':row['prediction_count'],'erasure':row['erasure']}),flush=True)
                del result
        for name,spec in specs.items():
            if source_identity(*spec)!=sources[name]:raise RuntimeError('source changed: '+name)
        if any(sha256(ROOT/'reproduction/cao2025'/k)!=v for k,v in original.items()):raise RuntimeError('original21 changed')
        info.update(state='passed',ended_at_unix=time.time(),detector_states_readonly=484,codec_states_readonly=768,
            native_DDP=True,distributed_backend='nccl',peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            prototype_sources=source_identity(HERE,['adapter.py','probe_liga_rgb_gpu.py']),protocol_sha256=sha256(HERE/'LIGA-native-GPU-engineering.md'),
            limitations='Only one engineeringframe; freshuntrained codec, noAP/fullfold/latency/learnedbaseline claim');save()
    except BaseException as exc:info.update(state='failed',exception=repr(exc),ended_at_unix=time.time());save();raise
    finally:
        for h in handles:h.remove()
        if dist.is_initialized():dist.destroy_process_group()
        if group is not None:group.cleanup()

if __name__=='__main__':main()
