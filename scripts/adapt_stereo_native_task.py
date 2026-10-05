#!/usr/bin/env python3
"""F6: frozen F5b student/receiver, existing16 codec parameters, clean native3D objective."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import numpy as np
import torch
import torch.distributed as dist

ROOT=Path(__file__).resolve().parents[1]
LIGA=ROOT/'third_party/LIGA-Stereo'
sys.path[:0]=[str(ROOT/'src'),str(LIGA),str(ROOT/'third_party/mmdetection_kitti')]
from geocomm.stereo_task_adaptation import NativeTaskAdaptation,SENSOR_TRAIN_KEYS,assert_frozen,freeze_except_codec
from geocomm.evidence import revision,sha256,source_identity
from geocomm.inference import SENSOR_INPUT_KEYS
from pretrain_student import fold_identity,write_json

INITIALIZATION_SHA='9a4b291070df160e6c03e75ed31eeee1dda7f6bfbaa0528a43a9cdf526977553'


def main():
    parser=argparse.ArgumentParser()
    for name in ('config','checkpoint','protocol','fold','source-manifest','output-dir','manifest'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--engineering-steps',type=int,choices=(0,3,6),default=0)
    args=parser.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path): setattr(args,key,value.resolve())
    if args.manifest.exists() or args.output_dir.exists() or args.engineering_steps<0:
        parser.error('unique outputs and nonnegative engineering step count required')
    if sha256(args.checkpoint)!=INITIALIZATION_SHA: parser.error('exact fixed F5b initialization required')
    usage=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    if len(usage)!=1:raise RuntimeError('locked singleGPU runtime required')
    gpu_used,gpu_total=(int(v.strip()) for v in usage[0].split(','))
    free_gpu,total_gpu=(gpu_total-gpu_used)*2**20,gpu_total*2**20
    if free_gpu<10*1024**3:raise RuntimeError('insufficient NVIDIA physical margin for original-training coexistence')
    args.output_dir.mkdir(parents=True);args.manifest.parent.mkdir(parents=True,exist_ok=True)
    source_specs={'project_sources':(ROOT,['src','scripts','configs','pyproject.toml']),
        'detector_sources':(LIGA,['liga','configs','tools','setup.py']),
        'mmdet_sources':(ROOT/'third_party/mmdetection_kitti',['mmdet']),
        'stereo_rcnn_sources':(ROOT/'third_party/Stereo-RCNN',['lib','demo.py','test_net.py'])}
    identities={name:source_identity(*spec) for name,spec in source_specs.items()}
    source=json.loads(args.source_manifest.read_text())
    for tree,key in [('project','project_sources'),('liga','detector_sources'),('mmdet','mmdet_sources'),('stereo_rcnn','stereo_rcnn_sources')]:
        if source[tree]['actual_sources']!=identities[key]: raise RuntimeError('frozen source differs: '+tree)
    info=dict(state='starting',evidence_type='engineering_only' if args.engineering_steps else 'exploratory_clean3D_codec_task_adaptation',
        seed=17,started_at_unix=time.time(),output_dir=str(args.output_dir),code_revision=revision(ROOT),
        protocol_sha256=sha256(args.protocol),config_sha256=sha256(args.config),fold_sha256=sha256(args.fold),
        initialization_sha256=sha256(args.checkpoint),source_manifest_sha256=sha256(args.source_manifest),
        torch=torch.__version__,cuda_runtime=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
        optimizer_steps=0,distributed_backend='nccl',gpu_free_before_bytes=free_gpu,gpu_total_bytes=total_gpu,**identities)
    write_json(args.manifest,info)
    group_directory=None;observer=None
    try:
        torch.multiprocessing.set_start_method('spawn',force=True)
        random.seed(17);np.random.seed(17);torch.manual_seed(17);torch.cuda.manual_seed_all(17)
        os.chdir(LIGA)
        from easydict import EasyDict
        from liga.config import cfg_from_yaml_file
        from liga.datasets import build_dataloader
        from liga.models import build_network
        from liga.utils.common_utils import create_logger
        config=cfg_from_yaml_file(str(args.config),EasyDict())
        opt=config.OPTIMIZATION;link_cfg=config.MODEL.BACKBONE_3D.STEREO_FEATURE_LINK
        if (opt.NUM_EPOCHS,opt.BATCH_SIZE_PER_GPU,opt.LR,opt.WEIGHT_DECAY,opt.GRAD_NORM_CLIP)!=(1,1,.0001,.0001,10):
            raise RuntimeError('F6 fixed optimizer/budget differs')
        if not (link_cfg.enabled and link_cfg.channel=='identity' and link_cfg.snr_db==10.
                and link_cfg.pilots==8 and not config.MODEL.BACKBONE_3D.SEMANTIC_LINK.enabled
                and config.MODEL.BACKBONE_3D.STUDENT_ENCODER.enabled
                and config.MODEL.BACKBONE_3D.STUDENT_ENCODER.distill_weight==0):
            raise RuntimeError('F6 fixed codec architecture differs')
        logger=create_logger(args.output_dir/'native_dataset.log')
        dataset,loader,_=build_dataloader(config.DATA_CONFIG,config.CLASS_NAMES,
            batch_size=1,dist=False,workers=4,training=True,logger=logger)
        fold=json.loads(args.fold.read_text());info['dataset']=fold_identity(dataset,fold)
        if len(loader)!=3340: raise RuntimeError('native one-epoch budget differs')
        group_directory=tempfile.TemporaryDirectory(prefix='F6-process-group-')
        dist.init_process_group('nccl',init_method='file://'+group_directory.name+'/rank',rank=0,world_size=1)
        model=build_network(config.MODEL,len(config.CLASS_NAMES),dataset).cuda().eval()
        initial=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        reference=initial['model_state'];actual=model.state_dict()
        trainable=freeze_except_codec(model);names=[name for name,_ in trainable];parameters=[p for _,p in trainable]
        if len(reference)!=535 or len(actual)!=535 or set(actual)!=set(reference) or len(names)!=16:
            raise RuntimeError('complete535-state F5b initialization and exact16 codec parameters required')
        if any(value.shape!=actual[name].shape or value.dtype!=actual[name].dtype or not torch.isfinite(value).all()
               for name,value in reference.items()):raise RuntimeError('initialization layout/dtype/finite differs')
        model.load_state_dict(reference,strict=True)
        if assert_frozen(reference,model.state_dict(),set())!=535:raise RuntimeError('initial values differ')
        initial_path=args.output_dir/'initialization.pth'
        torch.save(dict(model_state=reference,version='F6-exact-F5b-all535',seed=17),initial_path)
        info.update(full_initialization_path=str(initial_path),full_initialization_sha256=sha256(initial_path))
        if assert_frozen(reference,model.state_dict(),set(names))!=519:raise RuntimeError('F6 state scope differs')
        del initial,actual
        optimizer=torch.optim.AdamW([dict(params=parameters,parameter_names=names)],lr=.0001,weight_decay=.0001)
        observer=NativeTaskAdaptation(model)
        info.update(state='training',matched_initialization_tensors=535,full_state_tensors=535,frozen_state_tensors=519,
            trainable_parameter_names=names,training_inputs=sorted(SENSOR_TRAIN_KEYS),target_input='augmented_gt_boxes_at_native3D_head_only',
            original_teacher_calls_allowed=False,native_task_forward_allowed=True,all_modules_eval=True,
            budget=dict(epochs=1,train_frames=3340,updates=3340,holdout_frames=372,batch_size=1,workers=4,
                        lr=.0001,weight_decay=.0001,clip_norm=10,channel='identity',loss='native3D_classification_box_direction_no_teacher_depth_2D'),
            original_global_step_unchanged=True,peak_allocated_GiB_at_start=torch.cuda.max_memory_allocated()/2**30,holdout_limitation=fold['limitation'])
        write_json(args.manifest,info)
        returned_ids=[];steps=0;records=args.output_dir/'training.jsonl'
        with records.open('x') as stream:
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                sensors={key:batch[key] for key in SENSOR_TRAIN_KEYS if key in batch}
                for key in ('left_img','right_img','random_T'):
                    if key not in sensors:continue
                    sensors[key]=torch.as_tensor(sensors[key],device='cuda',dtype=torch.float32)
                targets=torch.as_tensor(batch['gt_boxes'],device='cuda',dtype=torch.float32)
                loss=observer.forward(sensors,targets)
                scalars=observer.pending
                if not torch.isfinite(loss): raise RuntimeError('nonfinite native3D task loss')
                if (scalars['total_complex_uses']!=62400 or scalars['channel']!='identity'
                        or scalars['stereo_complex_uses']!=49920 or scalars['appearance_complex_uses']!=12480
                        or scalars['boundary']!='stereo_features_before_receiver_cost'
                        or abs(scalars['tx_energy']/62400-1)>1e-4):
                    raise RuntimeError('fixed channel/energy budget differs')
                loss.backward()
                if any(p.grad is None for p in parameters): raise RuntimeError('missing codec gradient')
                if any(p.grad is not None for name,p in model.named_parameters() if name not in names):
                    raise RuntimeError('inactive parameter acquired a gradient')
                norm=torch.nn.utils.clip_grad_norm_(parameters,10.,error_if_nonfinite=True)
                if torch.cuda.max_memory_reserved()+2*1024**3>free_gpu:
                    raise RuntimeError('native task backward lacks2GiB NVIDIA coexistence margin before optimizer update')
                optimizer.step();steps+=1
                scalars=observer.finish_backward()
                frame=str(batch['frame_id'][0]);returned_ids.append(frame)
                stream.write(json.dumps(dict(step=steps,epoch=1,loss=float(loss.detach()),
                    preclip_grad_norm=float(norm),lr=.0001,**scalars),allow_nan=False)+'\n');stream.flush()
                info['optimizer_steps']=steps
                if steps==1 or steps%100==0:
                    write_json(args.manifest,info)
                    print(f'step={steps}/3340 loss={float(loss.detach()):.6f} grad={float(norm):.6f}',flush=True)
                if args.engineering_steps and steps>=args.engineering_steps: break
        if steps!=(args.engineering_steps or 3340) or not set(returned_ids)<=set(dataset.sample_id_list):
            raise RuntimeError('update/frame membership differs')
        calls=observer.calls
        if calls!=dict(steps=steps,student=2*steps,codec=steps,channel=steps,build_cost=steps,map_to_bev=steps,BEV=steps,head3D=steps,forbidden=0): raise RuntimeError('executed module coverage differs')
        frozen=assert_frozen(reference,model.state_dict(),set(names))
        for key,spec in source_specs.items():
            if source_identity(*spec)!=identities[key]: raise RuntimeError('source changed during training: '+key)
        checkpoint_path=args.output_dir/('checkpoint_sanity.pth' if args.engineering_steps else 'checkpoint_epoch_1.pth')
        torch.save(dict(model_state={name:value.detach().cpu() for name,value in model.state_dict().items()},
            optimizer_state=optimizer.state_dict(),epoch=0 if args.engineering_steps else 1,it=steps,
            version='F6-clean3D-codec-task-adaptation',rng_state=dict(python=random.getstate(),numpy=np.random.get_state(),
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())),checkpoint_path)
        info.update(state='finished',ended_at_unix=time.time(),checkpoint_path=str(checkpoint_path),
            checkpoint_sha256=sha256(checkpoint_path),training_records_sha256=sha256(records),
            frozen_states_identical=frozen,module_calls=calls,completed_epochs=int(not args.engineering_steps),
            unique_returned_training_frames=len(set(returned_ids)),native_empty_augmented_GT_resampling_retained=True,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
        if torch.cuda.max_memory_reserved()+2*1024**3>free_gpu:
            raise RuntimeError('measured full-task backward lacks2GiB NVIDIA coexistence margin')
        write_json(args.manifest,info);print(json.dumps({k:info[k] for k in ['state','optimizer_steps','frozen_states_identical','module_calls','checkpoint_sha256']}),flush=True)
    except BaseException as exc:
        info.update(state='failed',exception=repr(exc),ended_at_unix=time.time());write_json(args.manifest,info);raise
    finally:
        if observer is not None:observer.close()
        if dist.is_initialized(): dist.destroy_process_group()
        if group_directory is not None: group_directory.cleanup()


if __name__=='__main__': main()
