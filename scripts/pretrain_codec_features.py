#!/usr/bin/env python3
"""F4: exact frozen F3 student/receiver; sixteen codec tensors, one native epoch."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
import numpy as np
import torch
import torch.distributed as dist

ROOT=Path(__file__).resolve().parents[1]
LIGA=ROOT/'third_party/LIGA-Stereo'
sys.path[:0]=[str(ROOT/'src'),str(LIGA),str(ROOT/'third_party/mmdetection_kitti')]
from geocomm.codec_warmup import assert_frozen, capture_reconstruction, freeze_except_codec
from geocomm.evidence import revision,sha256,source_identity
from geocomm.inference import SENSOR_INPUT_KEYS
from pretrain_student import fold_identity,write_json

INITIALIZATION_SHA='cb50b4d4073fb2e5be31533d949a2141a46389436a5080429753de8b038aef7c'


def main():
    parser=argparse.ArgumentParser()
    for name in ('config','checkpoint','protocol','fold','source-manifest','output-dir','manifest'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--engineering-steps',type=int,default=0)
    args=parser.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path): setattr(args,key,value.resolve())
    if args.manifest.exists() or args.output_dir.exists() or args.engineering_steps<0:
        parser.error('unique outputs and nonnegative engineering step count required')
    if sha256(args.checkpoint)!=INITIALIZATION_SHA: parser.error('exact fixed F3 initialization required')
    args.output_dir.mkdir(parents=True);args.manifest.parent.mkdir(parents=True,exist_ok=True)
    source_specs={'project_sources':(ROOT,['src','scripts','configs','pyproject.toml']),
        'detector_sources':(LIGA,['liga','configs','tools','setup.py']),
        'mmdet_sources':(ROOT/'third_party/mmdetection_kitti',['mmdet'])}
    identities={name:source_identity(*spec) for name,spec in source_specs.items()}
    source=json.loads(args.source_manifest.read_text())
    for tree,key in [('project','project_sources'),('liga','detector_sources'),('mmdet','mmdet_sources')]:
        if source[tree]['actual_sources']!=identities[key]: raise RuntimeError('frozen source differs: '+tree)
    info=dict(state='starting',evidence_type='engineering_only' if args.engineering_steps else 'exploratory_codec_feature_warmup',
        seed=17,started_at_unix=time.time(),output_dir=str(args.output_dir),code_revision=revision(ROOT),
        protocol_sha256=sha256(args.protocol),config_sha256=sha256(args.config),fold_sha256=sha256(args.fold),
        initialization_sha256=sha256(args.checkpoint),source_manifest_sha256=sha256(args.source_manifest),
        torch=torch.__version__,cuda_runtime=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
        optimizer_steps=0,**identities)
    write_json(args.manifest,info)
    group_directory=None;guards=[]
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
        opt=config.OPTIMIZATION;link_cfg=config.MODEL.BACKBONE_3D.SEMANTIC_LINK
        if (opt.NUM_EPOCHS,opt.BATCH_SIZE_PER_GPU,opt.LR,opt.WEIGHT_DECAY,opt.GRAD_NORM_CLIP)!=(1,1,.0001,.0001,10):
            raise RuntimeError('F4 fixed optimizer/budget differs')
        if not (link_cfg.enabled and link_cfg.channel=='identity' and link_cfg.allocation=='uniform'
                and link_cfg.boundary=='raw_cost' and link_cfg.complex_width==4
                and tuple(link_cfg.geometry_stride)==(8,4,4) and link_cfg.appearance_stride==4
                and config.MODEL.BACKBONE_3D.STUDENT_ENCODER.enabled):
            raise RuntimeError('F4 fixed codec architecture differs')
        logger=create_logger(args.output_dir/'native_dataset.log')
        dataset,loader,_=build_dataloader(config.DATA_CONFIG,config.CLASS_NAMES,
            batch_size=1,dist=False,workers=4,training=True,logger=logger)
        fold=json.loads(args.fold.read_text());info['dataset']=fold_identity(dataset,fold)
        if len(loader)!=3340: raise RuntimeError('native one-epoch budget differs')
        group_directory=tempfile.TemporaryDirectory(prefix='F4-process-group-')
        dist.init_process_group('gloo',init_method='file://'+group_directory.name+'/rank',rank=0,world_size=1)
        model=build_network(config.MODEL,len(config.CLASS_NAMES),dataset).cuda().eval()
        initial=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        reference=initial['model_state'];actual=model.state_dict()
        if len(reference)!=549 or set(reference)!=set(actual): raise RuntimeError('complete549-state initialization required')
        if any(value.shape!=actual[name].shape or value.dtype!=actual[name].dtype for name,value in reference.items()):
            raise RuntimeError('initialization layout/dtype differs')
        model.load_state_dict(reference,strict=True)
        if assert_frozen(reference,model.state_dict(),set())!=549: raise RuntimeError('initial values differ')
        trainable=freeze_except_codec(model);names=[name for name,_ in trainable];parameters=[p for _,p in trainable]
        if len(names)!=16 or assert_frozen(reference,model.state_dict(),set(names))!=533:
            raise RuntimeError('F4 parameter scope differs')
        del initial,actual
        optimizer=torch.optim.AdamW([dict(params=parameters,parameter_names=names)],lr=.0001,weight_decay=.0001)
        calls=dict(student=0,codec=0,forbidden=0)
        def forbidden(module,inputs):
            calls['forbidden']+=1
            raise RuntimeError('teacher or downstream receiver executed during F4 warmup')
        for module in [model.backbone_3d.feature_backbone,model.backbone_3d.feature_neck,
                       model.lidar_model,model.backbone_3d.dres0,model.dense_head]:
            guards.append(module.register_forward_pre_hook(forbidden))
        def count_student(module,inputs): calls['student']+=1
        def count_codec(module,inputs): calls['codec']+=1
        guards.append(model.backbone_3d.student_semantic_link_encoder.register_forward_pre_hook(count_student))
        guards.append(model.backbone_3d.semantic_link.register_forward_pre_hook(count_codec))
        info.update(state='training',matched_initialization_tensors=549,frozen_state_tensors=533,
            trainable_parameter_names=names,training_inputs=list(SENSOR_INPUT_KEYS),
            original_teacher_calls_allowed=False,native_task_forward_allowed=False,
            budget=dict(epochs=1,train_frames=3340,updates=3340,holdout_frames=372,batch_size=1,workers=4,
                        lr=.0001,weight_decay=.0001,clip_norm=10,channel='identity',loss='cost_MSE_plus_appearance_MSE'),
            original_global_step_unchanged=True,holdout_limitation=fold['limitation'])
        write_json(args.manifest,info)
        returned_ids=[];steps=0;records=args.output_dir/'training.jsonl'
        with records.open('x') as stream:
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                sensors={key:batch[key] for key in SENSOR_INPUT_KEYS if key in batch}
                for key in ('left_img','right_img'):
                    sensors[key]=torch.as_tensor(sensors[key],device='cuda',dtype=torch.float32)
                loss,scalars=capture_reconstruction(model.backbone_3d,sensors)
                if not torch.isfinite(loss): raise RuntimeError('nonfinite reconstruction loss')
                if (scalars['total_complex_uses']!=62400 or scalars['channel']!='identity'
                        or abs(scalars['tx_energy']/62400-1)>1e-4):
                    raise RuntimeError('fixed channel/energy budget differs')
                loss.backward()
                if any(p.grad is None for p in parameters): raise RuntimeError('missing codec gradient')
                if any(p.grad is not None for name,p in model.named_parameters() if name not in names):
                    raise RuntimeError('inactive parameter acquired a gradient')
                norm=torch.nn.utils.clip_grad_norm_(parameters,10.,error_if_nonfinite=True)
                optimizer.step();steps+=1
                frame=str(batch['frame_id'][0]);returned_ids.append(frame)
                stream.write(json.dumps(dict(step=steps,epoch=1,frame_id=frame,loss=float(loss.detach()),
                    preclip_grad_norm=float(norm),lr=.0001,**scalars),allow_nan=False)+'\n');stream.flush()
                info['optimizer_steps']=steps
                if steps==1 or steps%100==0:
                    write_json(args.manifest,info)
                    print(f'step={steps}/3340 loss={float(loss.detach()):.6f} grad={float(norm):.6f}',flush=True)
                if args.engineering_steps and steps>=args.engineering_steps: break
        if steps!=(args.engineering_steps or 3340) or not set(returned_ids)<=set(dataset.sample_id_list):
            raise RuntimeError('update/frame membership differs')
        if calls!=dict(student=2*steps,codec=steps,forbidden=0): raise RuntimeError('executed module coverage differs')
        frozen=assert_frozen(reference,model.state_dict(),set(names))
        for key,spec in source_specs.items():
            if source_identity(*spec)!=identities[key]: raise RuntimeError('source changed during training: '+key)
        checkpoint_path=args.output_dir/('checkpoint_sanity.pth' if args.engineering_steps else 'checkpoint_epoch_1.pth')
        torch.save(dict(model_state={name:value.detach().cpu() for name,value in model.state_dict().items()},
            optimizer_state=optimizer.state_dict(),epoch=0 if args.engineering_steps else 1,it=steps,
            version='F4-codec-feature-only',rng_state=dict(python=random.getstate(),numpy=np.random.get_state(),
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())),checkpoint_path)
        info.update(state='finished',ended_at_unix=time.time(),checkpoint_path=str(checkpoint_path),
            checkpoint_sha256=sha256(checkpoint_path),training_records_sha256=sha256(records),
            frozen_states_identical=frozen,module_calls=calls,completed_epochs=int(not args.engineering_steps),
            unique_returned_training_frames=len(set(returned_ids)),native_empty_augmented_GT_resampling_retained=True)
        write_json(args.manifest,info);print(json.dumps({k:info[k] for k in ['state','optimizer_steps','frozen_states_identical','module_calls','checkpoint_sha256']}),flush=True)
    except BaseException as exc:
        info.update(state='failed',exception=repr(exc),ended_at_unix=time.time());write_json(args.manifest,info);raise
    finally:
        for guard in guards: guard.remove()
        if dist.is_initialized(): dist.destroy_process_group()
        if group_directory is not None: group_directory.cleanup()


if __name__=='__main__': main()
