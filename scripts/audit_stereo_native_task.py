#!/usr/bin/env python3
"""Independent F6 full-state, native3D loss, target-boundary and actualAdam audit."""
import argparse
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from geocomm.evidence import sha256


def check(condition,message):
    if not condition: raise ValueError(message)


def inactive_states(reference,actual,names):
    check(set(reference)==set(actual),'full state key coverage differs')
    unchanged=0;changed=[]
    for name,value in reference.items():
        other=actual[name]
        check(value.shape==other.shape and value.dtype==other.dtype,'state layout/dtype differs: '+name)
        check(torch.isfinite(other).all(),'nonfinite saved state: '+name)
        if name not in names:
            check(torch.equal(value,other),'inactive state differs: '+name);unchanged+=1
        elif not torch.equal(value,other): changed.append(name)
    return unchanged,changed


def scalar_records(path,ids,steps,iou_weight):
    losses=[];norms=[];frames=[];components={name:[] for name in ('classification_loss','location_loss','direction_loss','iou_loss_raw')}
    sequence=['student','student','link_start','channel','link_done','build_cost','backbone_done','map_to_bev','BEV','GT_at_3D_head','head3D']
    sensor_keys={'batch_size','left_img','right_img','calib','image_shape','frame_id'}
    native_gradient_shapes={'left_stereo':[1,32,320,1248],'right_stereo':[1,32,320,1248],
        'appearance':[1,32,80,312],'symbols':[1,62400,2],'native_cost':[1,64,72,80,312]}
    nonzero_gradient_frames={k:0 for k in native_gradient_shapes}
    empty_GT_frames=0
    for step,line in enumerate(path.read_text().splitlines(),1):
        r=json.loads(line)
        check(r['step']==step and r['epoch']==1,'noncontiguous update/epoch')
        check(r['frame_id'] in ids,'frame outside native training fold')
        for key in ('loss','classification_loss','location_loss','direction_loss','iou_loss_raw','native_reported_loss','preclip_grad_norm','tx_energy','lr','cbr'):
            check(math.isfinite(r[key]) and r[key]>=0,'invalid scalar: '+key)
        check(r['iou_weight']==iou_weight,'native IoU weighting differs')
        expected=r['classification_loss']+r['location_loss']+r['direction_loss']+iou_weight*r['iou_loss_raw']
        check(math.isclose(r['loss'],expected,rel_tol=3e-6,abs_tol=1e-6) and
              math.isclose(r['loss'],r['native_reported_loss'],rel_tol=3e-6,abs_tol=1e-6),'native3D loss reduction differs')
        check(r['loss_type']=='native3D_cls_box_direction' and r['lr']==.0001,'objective/LR differs')
        check(r['channel']=='identity' and r['allocation']=='uniform' and r['snr_db']==10.,'channel differs')
        check(r['data_complex_uses']==r['total_complex_uses']==62400 and r['pilot_complex_uses']==r['header_complex_uses']==0,'symbol budget differs')
        check(r['boundary']=='stereo_features_before_receiver_cost' and r['stereo_complex_uses']==49920 and r['appearance_complex_uses']==12480,'wire layout differs')
        check(abs(r['tx_energy']/62400-1)<=1e-4 and math.isclose(r['cbr'],1/38.4,rel_tol=1e-10),'energy/CBR differs')
        check(r['feature_shapes']=={k:v for k,v in native_gradient_shapes.items() if k in ('left_stereo','right_stereo','appearance')},'native feature interfaces differ')
        check(r['gradient_shapes']==native_gradient_shapes and set(r['gradient_norms'])==set(native_gradient_shapes),'native graph gradient coverage differs')
        check(all(math.isfinite(value) and value>=0 for value in r['gradient_norms'].values()),'missing/nonfinite native cost or feature gradient')
        check(r['sequence']==sequence and r['cost_inputs_are_received_features'] and r['appearance_input_is_received_feature'],'native receiver chronology or bypass differs')
        check(r['GT_introduced_only_at_head'] and r['all_modules_eval'],'GT/mode scope differs')
        check(len(r['GT_shape'])==3 and r['GT_shape'][0]==1 and r['GT_shape'][1]>=0 and r['GT_shape'][2]==8,'native augmentedGT shape differs')
        check(r['empty_GT']==(r['GT_shape'][1]==0),'emptyGT accounting differs')
        check(type(r['positive_anchors']) is int and r['positive_anchors']>=0 and
              type(r['background_anchors']) is int and r['background_anchors']>=0,'native anchor counts differ')
        if r['empty_GT']:
            empty_GT_frames+=1
            check(r['positive_anchors']==0 and r['background_anchors']>0 and
                  all(r[k]==0 for k in ('location_loss','direction_loss','iou_loss_raw')),
                  'emptyGT native background-only losses differ')
        keys=set(r['sender_input_keys']);check(keys==sensor_keys|({'random_T'} if r['random_T_present'] else set()),'sender input boundary differs')
        for name,norm in r['gradient_norms'].items():nonzero_gradient_frames[name]+=int(norm>0)
        losses.append(r['loss']);norms.append(r['preclip_grad_norm']);frames.append(r['frame_id'])
        for name in components:components[name].append(r[name])
    check(len(losses)==steps and steps>0,'raw update count differs')
    check(all(count>0 for count in nonzero_gradient_frames.values()),'entire run lacks a native task gradient branch')
    return dict(nonzero_gradient_frames=nonzero_gradient_frames,empty_GT_frames=empty_GT_frames,steps=steps,unique_returned_frames=len(set(frames)),max_preclip_gradient=max(norms),
        first100_loss_median=statistics.median(losses[:100]),last100_loss_median=statistics.median(losses[-100:]),
        component_first100_medians={k:statistics.median(v[:100]) for k,v in components.items()},
        component_last100_medians={k:statistics.median(v[-100:]) for k,v in components.items()})


def main():
    parser=argparse.ArgumentParser()
    for name in ('manifest','initialization','config','protocol','fold','source-manifest','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--engineering-sanity',action='store_true')
    parser.add_argument('--engineering-steps',type=int,choices=(3,6),default=3)
    args=parser.parse_args()
    if args.output.exists(): parser.error('retain previous evidence; choose unique output')
    run=json.loads(args.manifest.read_text());fold=json.loads(args.fold.read_text())
    check(run['state']=='finished','training not finished')
    check(run['evidence_type']==('engineering_only' if args.engineering_sanity else 'exploratory_clean3D_codec_task_adaptation'),'evidence scope differs')
    for path,key in [(args.initialization,'initialization_sha256'),(args.config,'config_sha256'),
                     (args.protocol,'protocol_sha256'),(args.fold,'fold_sha256'),(args.source_manifest,'source_manifest_sha256')]:
        check(sha256(path)==run[key],'input identity differs: '+key)
    check(run['initialization_sha256']=='9a4b291070df160e6c03e75ed31eeee1dda7f6bfbaa0528a43a9cdf526977553','fixed F5b predecessor differs')
    sources=json.loads(args.source_manifest.read_text())
    for tree,key in [('project','project_sources'),('liga','detector_sources'),('mmdet','mmdet_sources'),('stereo_rcnn','stereo_rcnn_sources')]:
        check(sources[tree]['actual_sources']==run[key],'source differs: '+tree)
    check(run['distributed_backend']=='nccl','nativeGPU loss collectives backend differs')
    check(run['seed']==17 and run['budget']==dict(epochs=1,train_frames=3340,updates=3340,holdout_frames=372,
        batch_size=1,workers=4,lr=.0001,weight_decay=.0001,clip_norm=10,channel='identity',loss='native3D_classification_box_direction_no_teacher_depth_2D'),'budget differs')
    steps=args.engineering_steps if args.engineering_sanity else 3340
    check(run['optimizer_steps']==steps and run['completed_epochs']==int(not args.engineering_sanity),'update/epoch budget differs')
    check(run['module_calls']==dict(steps=steps,student=2*steps,codec=steps,channel=steps,build_cost=steps,map_to_bev=steps,BEV=steps,head3D=steps,forbidden=0),'module execution count differs')
    check(set(run['training_inputs'])=={'batch_size','left_img','right_img','calib','image_shape','frame_id','random_T'},'feature input boundary differs')
    check(run['target_input']=='augmented_gt_boxes_at_native3D_head_only','target scope differs')
    check(run['peak_reserved_GiB']*2**30+2*1024**3<=run['gpu_free_before_bytes'],'measured NVIDIA coexistence margin failed')
    check(not run['original_teacher_calls_allowed'] and run['native_task_forward_allowed'] and run['all_modules_eval'],'teacher/task scope differs')
    checkpoint=Path(run['checkpoint_path']);check(sha256(checkpoint)==run['checkpoint_sha256'],'checkpoint identity differs')
    actual=torch.load(checkpoint,map_location='cpu',weights_only=False)
    predecessor=torch.load(args.initialization,map_location='cpu',weights_only=False)['model_state']
    initial_path=Path(run['full_initialization_path'])
    check(sha256(initial_path)==run['full_initialization_sha256'],'complete initialization identity differs')
    reference=torch.load(initial_path,map_location='cpu',weights_only=False)['model_state']
    check(len(predecessor)==len(reference)==len(actual['model_state'])==535,'complete535-state coverage differs')
    check(set(predecessor)==set(reference),'completeF5b initialization keys differ')
    for name,value in predecessor.items():
        check(value.dtype==reference[name].dtype and value.shape==reference[name].shape
              and torch.equal(value,reference[name]),'initialization changed F5b: '+name)
    expected=set()
    for branch,indices in {'stereo_encoder':[1,3],'stereo_decoder':[0,2],
                           'appearance_encoder':[1,3],'appearance_decoder':[0,2]}.items():
        for index in indices:
            expected.update('backbone_3d.stereo_feature_link.'+branch+'.'+str(index)+'.'+field for field in ('weight','bias'))
    names=run['trainable_parameter_names'];check(len(names)==16 and set(names)==expected,'exact16-parameter scope differs')
    unchanged,changed=inactive_states(reference,actual['model_state'],expected)
    check(unchanged==run['frozen_states_identical']==519 and set(changed)==expected,'inactive state or learned parameter coverage differs')
    check(actual['epoch']==(0 if args.engineering_sanity else 1) and actual['it']==steps,'checkpoint counters differ')
    opt=actual['optimizer_state'];groups=opt['param_groups']
    check(len(groups)==1,'optimizer group count differs');group=groups[0]
    check(group['lr']==.0001 and group['weight_decay']==.0001 and group['betas']==(.9,.999) and group['eps']==1e-8,'AdamW settings differ')
    check(group['parameter_names']==names and len(group['params'])==16 and len(set(group['params']))==16
          and set(group['params'])==set(opt['state']),'optimizer parameter map differs')
    for name,key in zip(names,group['params']):
        state=opt['state'][key];parameter=actual['model_state'][name]
        check(float(state['step'])==steps,'actual optimizer update count differs: '+name)
        for moment in ('exp_avg','exp_avg_sq'):
            check(state[moment].shape==parameter.shape and torch.isfinite(state[moment]).all(),'invalid moment: '+name)
        check((state['exp_avg_sq']>=0).all() and (state['exp_avg_sq']>0).any(),'missing recorded learning signal: '+name)
    data=run['dataset'];train=fold['folds']['geocomm_tune_train'];root=Path(data['root'])
    check(data['split']=='geocomm_tune_train' and data['count']==3340 and len(train['ids'])==3340,'training fold differs')
    check(data['split_sha256']==train['split_sha256']==sha256(root/'ImageSets/geocomm_tune_train.txt'),'training IDs identity differs')
    check((root/'ImageSets/geocomm_tune_train.txt').read_text().split()==train['ids'],'training IDs/order differs')
    check(data['infos']['kitti_infos_geocomm_tune_train.pkl']==train['infos_sha256']==sha256(root/'kitti_infos_geocomm_tune_train.pkl'),'training infos identity differs')
    check(data['download_manifest_sha256']==sha256(root/'download-manifest.json'),'archive manifest differs')
    check(not set(train['ids'])&set(fold['folds']['geocomm_tune_holdout']['ids']),'fold overlap')
    records=Path(run['output_dir'])/'training.jsonl';check(sha256(records)==run['training_records_sha256'],'raw records identity differs')
    snapshot=args.output.with_suffix('.records');snapshot.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(records,snapshot/'training.jsonl')
    check(sha256(snapshot/'training.jsonl')==sha256(records),'snapshot identity differs')
    config=yaml.safe_load(args.config.read_text())
    iou_weight=config['MODEL']['DENSE_HEAD']['LOSS_CONFIG']['LOSS_WEIGHTS']['iou_weight']
    summary=scalar_records(snapshot/'training.jsonl',set(train['ids']),steps,iou_weight)
    if args.engineering_sanity and steps==6:
        check(summary['empty_GT_frames']>0,'repaired engineering did not exercise native emptyGT')
    check(summary['unique_returned_frames']==run['unique_returned_training_frames'],'returned frame count differs')
    result=dict(state='passed',evidence_type=run['evidence_type'],steps=steps,inactive_states_identical=519,
        trained_parameter_tensors=16,all_optimizer_update_counts=steps,summary=summary,
        full_initialization_sha256=run['full_initialization_sha256'],
        checkpoint_sha256=run['checkpoint_sha256'],manifest_sha256=sha256(args.manifest),
        source_manifest_sha256=sha256(args.source_manifest),raw_snapshot_sha256=sha256(snapshot/'training.jsonl'),
        limitations='exact state and saved scalar/counter audit; no fresh feature reconstruction or detection AP proof')
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))


if __name__=='__main__': main()
