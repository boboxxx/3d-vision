#!/usr/bin/env python3
"""Independent F5 complete checkpoint, exact inactive states and scalar audit of stereo feature transport."""
import argparse
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
import torch

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


def scalar_records(path,ids,steps):
    losses=[];norms=[];frames=[];costs=[];appearances=[]
    for step,line in enumerate(path.read_text().splitlines(),1):
        r=json.loads(line)
        check(r['step']==step and r['epoch']==1,'noncontiguous update/epoch')
        check(r['frame_id'] in ids,'frame outside native training fold')
        for key in ('loss','left_stereo_mse','right_stereo_mse','appearance_mse','preclip_grad_norm','tx_energy','lr','cbr'):
            check(math.isfinite(r[key]) and r[key]>=0,'invalid scalar: '+key)
        check(math.isclose(r['loss'],.5*(r['left_stereo_mse']+r['right_stereo_mse'])+r['appearance_mse'],rel_tol=2e-6,abs_tol=1e-7),'loss reduction differs')
        check(r['lr']==.0001 and r['channel']=='identity' and r['allocation']=='uniform','LR/channel/allocation differs')
        check(r['data_complex_uses']==r['total_complex_uses']==62400 and
              r['pilot_complex_uses']==r['header_complex_uses']==0 and r['snr_db']==10.,'symbol accounting differs')
        check(abs(r['tx_energy']/62400-1)<=1e-4 and math.isclose(r['cbr'],1/38.4,rel_tol=1e-10),'energy/CBR differs')
        check(r['left_stereo_shape']==r['right_stereo_shape']==[1,32,320,1248] and r['appearance_shape']==[1,32,80,312],'native feature shapes differ')
        check(r['boundary']=='stereo_features_before_receiver_cost' and r['stereo_complex_uses']==49920
              and r['appearance_complex_uses']==12480,'stereo feature wire layout differs')
        losses.append(r['loss']);norms.append(r['preclip_grad_norm']);frames.append(r['frame_id'])
        costs.append(.5*(r['left_stereo_mse']+r['right_stereo_mse']));appearances.append(r['appearance_mse'])
    check(len(losses)==steps and steps>0,'raw scalar update count differs')
    return dict(steps=steps,unique_returned_frames=len(set(frames)),max_preclip_gradient=max(norms),
        first100_loss_median=statistics.median(losses[:100]),last100_loss_median=statistics.median(losses[-100:]),
        first100_stereo_mean_median=statistics.median(costs[:100]),last100_stereo_mean_median=statistics.median(costs[-100:]),
        first100_appearance_median=statistics.median(appearances[:100]),last100_appearance_median=statistics.median(appearances[-100:]))


def main():
    parser=argparse.ArgumentParser()
    for name in ('manifest','initialization','config','protocol','fold','source-manifest','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--engineering-sanity',action='store_true')
    args=parser.parse_args()
    if args.output.exists(): parser.error('retain previous evidence; choose unique output')
    run=json.loads(args.manifest.read_text());fold=json.loads(args.fold.read_text())
    check(run['state']=='finished','training not finished')
    check(run['evidence_type']==('engineering_only' if args.engineering_sanity else 'exploratory_stereo_feature_warmup'),'evidence scope differs')
    for path,key in [(args.initialization,'initialization_sha256'),(args.config,'config_sha256'),
                     (args.protocol,'protocol_sha256'),(args.fold,'fold_sha256'),(args.source_manifest,'source_manifest_sha256')]:
        check(sha256(path)==run[key],'input identity differs: '+key)
    check(run['initialization_sha256']=='7e1ceacb4abff0bdb32ef89507b9786e1d80ae9c166ccaaeb4ffb4fcbd639bd2','fixed F2 predecessor differs')
    sources=json.loads(args.source_manifest.read_text())
    for tree,key in [('project','project_sources'),('liga','detector_sources'),('mmdet','mmdet_sources'),('stereo_rcnn','stereo_rcnn_sources')]:
        check(sources[tree]['actual_sources']==run[key],'source differs: '+tree)
    check(run['seed']==17 and run['budget']==dict(epochs=1,train_frames=3340,updates=3340,holdout_frames=372,
        batch_size=1,workers=4,lr=.0001,weight_decay=.0001,clip_norm=10,channel='identity',loss='half_left_right_stereo_MSE_plus_appearance_MSE'),'budget differs')
    steps=3 if args.engineering_sanity else 3340
    check(run['optimizer_steps']==steps and run['completed_epochs']==int(not args.engineering_sanity),'update/epoch budget differs')
    check(run['module_calls']==dict(student=2*steps,codec=steps,forbidden=0),'module execution count differs')
    check(set(run['training_inputs'])=={'batch_size','left_img','right_img','calib','image_shape','frame_id'},'feature input boundary differs')
    check(not run['original_teacher_calls_allowed'] and not run['native_task_forward_allowed'],'teacher/task scope differs')
    checkpoint=Path(run['checkpoint_path']);check(sha256(checkpoint)==run['checkpoint_sha256'],'checkpoint identity differs')
    actual=torch.load(checkpoint,map_location='cpu',weights_only=False)
    predecessor=torch.load(args.initialization,map_location='cpu',weights_only=False)['model_state']
    initial_path=Path(run['full_initialization_path'])
    check(sha256(initial_path)==run['full_initialization_sha256'],'complete initialization identity differs')
    reference=torch.load(initial_path,map_location='cpu',weights_only=False)['model_state']
    check(len(predecessor)==519 and len(reference)==len(actual['model_state'])==535,'complete535-state coverage differs')
    expected=set();shapes={}
    for branch,indices in {'stereo_encoder':[1,3],'stereo_decoder':[0,2],
                           'appearance_encoder':[1,3],'appearance_decoder':[0,2]}.items():
        widths=([32,32,2] if branch=='stereo_encoder' else [2,32,32] if branch=='stereo_decoder'
                else [32,32,8] if branch=='appearance_encoder' else [8,32,32])
        kernel=1 if branch.endswith('encoder') else 3
        for j,index in enumerate(indices):
            prefix='backbone_3d.stereo_feature_link.'+branch+'.'+str(index)+'.'
            shapes[prefix+'weight']=(widths[j+1],widths[j],kernel,kernel)
            shapes[prefix+'bias']=(widths[j+1],)
            expected.update([prefix+'weight',prefix+'bias'])
    check(set(reference)-set(predecessor)==expected and set(predecessor)<=set(reference),'exact fresh-state keys differ')
    for name,value in predecessor.items():
        check(value.dtype==reference[name].dtype and torch.equal(value,reference[name]),'full initialization changed F2: '+name)
    for name,shape in shapes.items():
        check(tuple(reference[name].shape)==shape and reference[name].dtype==torch.float32
              and torch.isfinite(reference[name]).all(),'fresh codec state layout/finite differs: '+name)
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
    summary=scalar_records(snapshot/'training.jsonl',set(train['ids']),steps)
    check(summary['unique_returned_frames']==run['unique_returned_training_frames'],'returned frame count differs')
    result=dict(state='passed',evidence_type=run['evidence_type'],steps=steps,inactive_states_identical=519,
        trained_parameter_tensors=16,all_optimizer_update_counts=steps,summary=summary,
        full_initialization_sha256=run['full_initialization_sha256'],
        checkpoint_sha256=run['checkpoint_sha256'],manifest_sha256=sha256(args.manifest),
        source_manifest_sha256=sha256(args.source_manifest),raw_snapshot_sha256=sha256(snapshot/'training.jsonl'),
        limitations='exact state and saved scalar/counter audit; no fresh feature reconstruction or detection AP proof')
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))


if __name__=='__main__': main()
