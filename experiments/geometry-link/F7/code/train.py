"""Prelocked matched F7 native codec adaptation; old F6 code/protocol untouched."""
import argparse,json,os,random,subprocess,sys,tempfile,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];LIGA=ROOT/'third_party/LIGA-Stereo'
sys.path[:0]=[str(HERE),str(ROOT/'src'),str(ROOT/'scripts'),str(LIGA),str(ROOT/'third_party/mmdetection_kitti')]
from geocomm.stereo_task_adaptation import SENSOR_TRAIN_KEYS,assert_frozen
from geocomm.evidence import revision,sha256,source_identity
from pretrain_student import fold_identity,write_json
from conditions import PARENT_SHA,PROTOCOL_SHA,CONFIG_SHA,STEPS,PairedChannel,augmented_evidence,check,schedule_manifest
from native_observer import MatchedNativeTaskAdaptation,freeze_matched_codec

def source_specs():
    return {'project':(ROOT,['src','scripts','configs','pyproject.toml']),
      'liga':(LIGA,['liga','configs','tools','setup.py']),
      'mmdet':(ROOT/'third_party/mmdetection_kitti',['mmdet']),
      'stereo_rcnn':(ROOT/'third_party/Stereo-RCNN',['lib','demo.py','test_net.py']),
      'experiment':(ROOT,['experiments/geometry-link/F7/code'])}

def main():
    p=argparse.ArgumentParser()
    for name in ('config','checkpoint','protocol','fold','source-manifest','output-dir','manifest'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--arm',choices=('identity','awgn'),required=True)
    p.add_argument('--engineering-steps',type=int,choices=(0,6),default=0);args=p.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.resolve())
    if args.manifest.exists() or args.output_dir.exists():p.error('unique outputs required')
    check(sha256(args.checkpoint)==PARENT_SHA,'sole full F6b parent required')
    check(sha256(args.protocol)==PROTOCOL_SHA and sha256(args.config)==CONFIG_SHA,'exact prelocked protocol/native config required')
    closure=json.loads((ROOT/'data/provenance/stereo-native-task-seed17-002-closure.json').read_text())
    check(closure['state']=='closed_all_audits_passed' and closure['checkpoint_sha256']==PARENT_SHA,'F6b closure required')
    # The isolated detector probe is pending in the current session. Never start
    # new GPU work before that native probe has reached its audited endpoint.
    probe=json.loads((ROOT/'data/engineering/original-StereoRCNN-RGB-GPU-endpoint-probe-001.json').read_text())
    check(probe['state']=='passed' and probe['detector_states_readonly']==670 and probe['codec_states_readonly']==768,'terminal Stereo-RCNN engineering required')
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    check(len(memory)==1,'single GPU required');used,total=(int(v.strip()) for v in memory[0].split(','));free=(total-used)*2**20
    check(free>=10*2**30,'NVIDIA physical margin insufficient; do not disturb original job')
    originals=json.loads((ROOT/'data/provenance/cao2025-staged-trainer-source-bfee9d8.json').read_text())
    check(len(originals)==21 and all(sha256(ROOT/'reproduction/cao2025'/k)==v for k,v in originals.items()),'original21 sources changed')
    check(sha256(ROOT/'reproduction/cao2025/formal-protocol.md')=='682fb774e09b9be7f68d09f8251dac6d4b99af3290f02660951d398f9e00eace','original protocol changed')
    identities={k:source_identity(*v) for k,v in source_specs().items()};source=json.loads(args.source_manifest.read_text())
    check(set(source)==set(identities),'complete five-tree source manifest')
    check(all(source[k]['actual_sources']==v for k,v in identities.items()),'frozen source differs')
    args.output_dir.mkdir(parents=True);args.manifest.parent.mkdir(parents=True,exist_ok=True)
    info=dict(state='starting',evidence_type='engineering_only' if args.engineering_steps else 'exploratory_matched_channel_native3D_codec_adaptation',
      arm=args.arm,seed=17,started_at_unix=time.time(),output_dir=str(args.output_dir),code_revision=revision(ROOT),
      protocol_sha256=sha256(args.protocol),config_sha256=sha256(args.config),fold_sha256=sha256(args.fold),
      initialization_sha256=sha256(args.checkpoint),source_manifest_sha256=sha256(args.source_manifest),source_identities=identities,
      optimizer_steps=0,distributed_backend='nccl',gpu_free_before_bytes=free,gpu_total_bytes=total*2**20,
      original21_sources=originals,schedule=schedule_manifest(),torch=torch.__version__,cuda_runtime=torch.version.cuda)
    write_json(args.manifest,info);group=None;observer=None;condition=None
    try:
        torch.multiprocessing.set_start_method('spawn',force=True)
        random.seed(17);np.random.seed(17);torch.manual_seed(17);torch.cuda.manual_seed_all(17);torch.backends.cudnn.benchmark=False
        os.chdir(LIGA)
        from easydict import EasyDict
        from liga.config import cfg_from_yaml_file
        from liga.datasets import build_dataloader
        from liga.models import build_network
        from liga.utils.common_utils import create_logger
        config=cfg_from_yaml_file(str(args.config),EasyDict());opt=config.OPTIMIZATION;link_cfg=config.MODEL.BACKBONE_3D.STEREO_FEATURE_LINK
        check((opt.NUM_EPOCHS,opt.BATCH_SIZE_PER_GPU,opt.LR,opt.WEIGHT_DECAY,opt.GRAD_NORM_CLIP)==(1,1,.0001,.0001,10),'fixed optimizer')
        check(link_cfg.enabled and link_cfg.channel=='identity' and link_cfg.snr_db==10. and link_cfg.pilots==8
              and not config.MODEL.BACKBONE_3D.SEMANTIC_LINK.enabled and config.MODEL.BACKBONE_3D.STUDENT_ENCODER.enabled
              and config.MODEL.BACKBONE_3D.STUDENT_ENCODER.distill_weight==0,'unaltered F6b architecture config required')
        # Explicitly registered CLI treatment overrides channel before model build.
        # There is no learned/checkpoint field for channel policy.
        link_cfg.channel=args.arm
        dataset,loader,_=build_dataloader(config.DATA_CONFIG,config.CLASS_NAMES,batch_size=1,dist=False,workers=4,training=True,logger=create_logger(args.output_dir/'native_dataset.log'))
        fold=json.loads(args.fold.read_text());info['dataset']=fold_identity(dataset,fold);check(len(loader)==STEPS,'native full epoch')
        group=tempfile.TemporaryDirectory(prefix='F7-process-group-');dist.init_process_group('nccl',init_method='file://'+group.name+'/rank',rank=0,world_size=1)
        model=build_network(config.MODEL,len(config.CLASS_NAMES),dataset).cuda().eval()
        reference=torch.load(args.checkpoint,map_location='cpu',weights_only=False)['model_state'];actual=model.state_dict()
        trainable=freeze_matched_codec(model,args.arm);names=[n for n,_ in trainable];parameters=[v for _,v in trainable]
        check(len(reference)==len(actual)==535 and set(reference)==set(actual) and len(names)==16,'535-state/16 parameter scope')
        check(all(v.shape==actual[k].shape and v.dtype==actual[k].dtype and torch.isfinite(v).all() for k,v in reference.items()),'parent dtype/layout/finite')
        model.load_state_dict(reference,strict=True);check(assert_frozen(reference,model.state_dict(),set())==535,'whole parent load')
        initial_path=args.output_dir/'initialization.pth';torch.save(dict(model_state=reference,version='F7-exact-F6b-all535',seed=17),initial_path)
        check(assert_frozen(reference,model.state_dict(),set(names))==519,'frozen scope');del actual
        optimizer=torch.optim.AdamW([dict(params=parameters,parameter_names=names)],lr=.0001,weight_decay=.0001)
        observer=MatchedNativeTaskAdaptation(model,args.arm);condition=PairedChannel(model.backbone_3d.stereo_feature_link,args.arm,'cuda')
        info.update(state='training',full_initialization_path=str(initial_path),full_initialization_sha256=sha256(initial_path),
          full_state_tensors=535,frozen_state_tensors=519,trainable_parameter_names=names,training_inputs=sorted(SENSOR_TRAIN_KEYS),
          target_input='augmented_gt_boxes_at_native3D_head_only',all_modules_eval=True,original_teacher_calls_allowed=False,
          native_task_forward_allowed=True,budget=dict(epochs=1,train_frames=STEPS,updates=STEPS,holdout_frames=372,batch_size=1,workers=4,
            lr=.0001,weight_decay=.0001,clip_norm=10,channel=args.arm,SNR_uniform_dB=[0.,20.],SNR_seed=1707,noise_seed=1708,
            loss='native3D_classification_box_direction_no_teacher_depth_2D'),holdout_limitation=fold['limitation'])
        write_json(args.manifest,info);steps=0;ids=[];records=args.output_dir/'training.jsonl'
        with records.open('x') as stream:
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                sensors={k:batch[k] for k in SENSOR_TRAIN_KEYS if k in batch}
                for k in ('left_img','right_img','random_T'):
                    if k in sensors:sensors[k]=torch.as_tensor(sensors[k],dtype=torch.float32,device='cuda')
                targets=torch.as_tensor(batch['gt_boxes'],dtype=torch.float32,device='cuda')
                fingerprint=augmented_evidence(sensors,targets);condition.prepare(steps+1)
                loss=observer.forward(sensors,targets);scalars=observer.pending
                check(scalars['total_complex_uses']==62400 and scalars['channel']==args.arm
                  and scalars['snr_db']==condition.schedule['SNR_values'][steps] and scalars['stereo_complex_uses']==49920
                  and scalars['appearance_complex_uses']==12480 and scalars['pilot_complex_uses']==scalars['header_complex_uses']==0
                  and scalars['boundary']=='stereo_features_before_receiver_cost' and abs(scalars['tx_energy']/62400-1)<=1e-4,'actual channel layout/SNR/energy')
                loss.backward();check(all(v.grad is not None for v in parameters),'missing codec gradient')
                check(all(v.grad is None for k,v in model.named_parameters() if k not in names),'inactive gradient')
                norm=torch.nn.utils.clip_grad_norm_(parameters,10.,error_if_nonfinite=True)
                check(torch.cuda.max_memory_reserved()+2*2**30<=free,'physical margin before actual optimizer step')
                optimizer.step();steps+=1;scalars=observer.finish_backward()
                check(augmented_evidence(sensors,targets)==fingerprint,'native forward mutated sensor/GT inputs')
                row=dict(step=steps,epoch=1,loss=float(loss.detach()),preclip_grad_norm=float(norm),lr=.0001,
                         channel_condition=condition.last_evidence,augmented_evidence=fingerprint,**scalars)
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();ids.append(row['frame_id'])
                if steps%100==0:
                    info['optimizer_steps']=steps;write_json(args.manifest,info);print(json.dumps(dict(step=steps,arm=args.arm,frame_id=row['frame_id'],loss=row['loss'])),flush=True)
                del loss,targets,sensors,batch,scalars,row
                if args.engineering_steps and steps==args.engineering_steps:break
        check(steps==(args.engineering_steps or STEPS) and condition.calls==steps,'fixed actual budget')
        calls=dict(steps=steps,student=2*steps,codec=steps,channel=steps,build_cost=steps,map_to_bev=steps,BEV=steps,head3D=steps,forbidden=0)
        check(observer.calls==calls,'actual native module chronology coverage')
        frozen=assert_frozen(reference,model.state_dict(),set(names))
        check(all(source_identity(*v)==identities[k] for k,v in source_specs().items()),'source changed during training')
        checkpoint=args.output_dir/('checkpoint_sanity.pth' if args.engineering_steps else 'checkpoint_epoch_1.pth')
        torch.save(dict(model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer_state=optimizer.state_dict(),
          epoch=0 if args.engineering_steps else 1,it=steps,version='F7-matched-native3D-codec-adaptation',arm=args.arm,
          channel_rng=condition.checkpoint_rng(),schedule=condition.schedule,
          rng_state=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())),checkpoint)
        info.update(state='finished',ended_at_unix=time.time(),optimizer_steps=steps,completed_epochs=int(not args.engineering_steps),
          checkpoint_path=str(checkpoint),checkpoint_sha256=sha256(checkpoint),training_records_sha256=sha256(records),
          frozen_states_identical=frozen,module_calls=calls,unique_returned_training_frames=len(set(ids)),
          peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
        write_json(args.manifest,info);print(json.dumps({k:info[k] for k in ('state','arm','optimizer_steps','checkpoint_sha256')}),flush=True)
    except BaseException as exc:
        info.update(state='failed',optimizer_steps=locals().get('steps',0),exception=repr(exc),ended_at_unix=time.time());write_json(args.manifest,info);raise
    finally:
        if observer is not None:observer.close()
        if condition is not None:condition.close()
        if dist.is_initialized():dist.destroy_process_group()
        if group is not None:group.cleanup()

if __name__=='__main__':main()
