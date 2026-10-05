"""Independent F9 saved-state/actualAdam/native scalar/RNG audit; no model forward."""
import argparse,json,math,shutil,statistics,sys
from pathlib import Path
import torch
import yaml
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path[:0]=[str(HERE),str(ROOT/'src'),str(ROOT/'scripts')]
from geocomm.evidence import sha256,source_identity
from audit_stereo_native_task import inactive_states
from conditions import PARENT_SHA,PROTOCOL_SHA,CONFIG_SHA,STEPS,check,schedule_manifest,verify_augmented_evidence,replay_rng_checkpoint

SHAPES={'left_stereo':[1,32,320,1248],'right_stereo':[1,32,320,1248],
        'appearance':[1,32,80,312],'symbols':[1,62400,2],'native_cost':[1,64,72,80,312]}
SEQUENCE=['student','student','link_start','channel','link_done','build_cost','backbone_done','map_to_bev','BEV','GT_at_3D_head','head3D']

def scalar_records(path,ids,steps,iou_weight,arm):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines()]
    check(len(rows)==steps and steps in (6,STEPS) and arm in ('U','G','P','S'),'fixed raw scope')
    schedule=schedule_manifest();nonzero={k:0 for k in SHAPES};empty=0;frames=[];losses=[];prior=None
    for step,r in enumerate(rows,1):
        check(r['step']==step and r['epoch']==1 and r['frame_id'] in ids,'native step/ID scope')
        for k in ('loss','classification_loss','location_loss','direction_loss','iou_loss_raw','native_reported_loss','preclip_grad_norm','tx_energy','lr','cbr'):
            check(math.isfinite(r[k]) and r[k]>=0,'nonfinite/negative native scalar: '+k)
        expected=r['classification_loss']+r['location_loss']+r['direction_loss']+iou_weight*r['iou_loss_raw']
        check(r['iou_weight']==iou_weight and math.isclose(r['loss'],expected,rel_tol=3e-6,abs_tol=1e-6)
              and math.isclose(r['loss'],r['native_reported_loss'],rel_tol=3e-6,abs_tol=1e-6),'native weighted loss')
        check(r['loss_type']=='native3D_cls_box_direction' and r['lr']==.0001,'objective/LR')
        check(r['channel']=='awgn' and r['snr_db']==schedule['SNR_values'][step-1] and r['allocation']==('uniform' if arm=='U' else 'F9_'+arm),'actual fixed arm/SNR')
        check(r['data_complex_uses']==r['total_complex_uses']==62400 and r['pilot_complex_uses']==r['header_complex_uses']==0
              and r['stereo_complex_uses']==49920 and r['appearance_complex_uses']==12480,'physical budget')
        check(r['boundary']=='stereo_features_before_receiver_cost' and abs(r['tx_energy']/62400-1)<=1e-4
              and math.isclose(r['cbr'],1/38.4,rel_tol=1e-10),'wire boundary/energy/CBR')
        check(r['feature_shapes']=={k:v for k,v in SHAPES.items() if k in ('left_stereo','right_stereo','appearance')}
              and r['gradient_shapes']==SHAPES and set(r['gradient_norms'])==set(SHAPES),'native shape/graph coverage')
        check(all(math.isfinite(v) and v>=0 for v in r['gradient_norms'].values()),'finite received/cost/symbol gradients')
        check(r['sequence']==SEQUENCE and r['cost_inputs_are_received_features'] and r['appearance_input_is_received_feature']
              and r['GT_introduced_only_at_head'] and r['all_modules_eval'],'native chronology/mode/target barrier')
        gt=r['GT_shape'];check(len(gt)==3 and gt[0]==1 and type(gt[1]) is int and gt[1]>=0 and gt[2]==8,'native GT layout')
        check(r['empty_GT']==(gt[1]==0) and type(r['positive_anchors']) is int and r['positive_anchors']>=0
              and type(r['background_anchors']) is int and r['background_anchors']>=0,'native empty/anchor accounting')
        if gt[1]==0:
            empty+=1;check(r['positive_anchors']==0 and r['background_anchors']>0
              and all(r[k]==0 for k in ('location_loss','direction_loss','iou_loss_raw')),'background-only loss semantics')
        keys={'batch_size','left_img','right_img','calib','image_shape','frame_id'}|({'random_T'} if r['random_T_present'] else set())
        check(set(r['sender_input_keys'])==keys,'GT/private data at sender')
        fp=r['augmented_evidence'];verify_augmented_evidence(fp)
        check(fp['frame_id']==r['frame_id'] and fp['arrays']['gt_boxes']['shape']==gt,'augmented evidence target/ID')
        for k in ('left_img','right_img'):
            check(fp['arrays'][k]['shape']==[1,3,320,1248] and fp['arrays'][k]['dtype']=='<f4','native actual sensor shape/dtype')
        check(fp['arrays']['gt_boxes']['dtype']=='<f4' and ('random_T' in fp['arrays'])==r['random_T_present'],'actual GT/transform scope')
        e=r['channel_condition'];check(e['step']==step and e['snr_db']==r['snr_db'] and e['channel']=='awgn'
          and e['schedule_sha256']==schedule['sha256'] and e['noise_seed']==1928 and e['isolated_noise_generator'] is True,'noise condition evidence')
        if prior is not None:check(e['noise_rng_before_sha256']==prior,'noise RNG continuity')
        check((e['noise_rng_before_sha256']==e['noise_rng_after_sha256'])==False,'actual RNG consumption')
        prior=e['noise_rng_after_sha256']
        coding=r['F9_coding']
        check(coding['arm']==arm and coding['nominal_snr']==r['snr_db'] and coding['data_complex_uses']==62400 and coding['head_parameters']==561 and not coding['receiver_side_information'],'F9 public sender/receiver contract')
        energy=coding['stereo_group_energy'];gains=coding['amplitude_gains']
        check(len(energy)==len(gains)==64 and coding['group_energy_layout']=='left32_then_right32_pixel_center_groups','complete public group records')
        check(all(math.isfinite(v) and v>=0 for v in energy) and math.isfinite(coding['appearance_energy']) and coding['appearance_energy']>=0,'actual finite group energy')
        check(all(math.isfinite(v) and .5<=v<=2 for v in gains),'bounded actual amplitude gains')
        check(math.isclose(sum(energy)+coding['appearance_energy'],coding['actual_energy_float64'],rel_tol=1e-12,abs_tol=1e-8),'all64 groups plus appearance charge actual total')
        check(abs(coding['actual_energy_float64']/62400-1)<=1e-5 and math.isclose(coding['actual_energy_float64'],r['tx_energy'],rel_tol=1e-5,abs_tol=1e-3),'actual normalization and channel accounting agree')
        if arm=='U':check(gains==[1.]*64,'uniform bypass')
        if step==1:check(gains==[1.]*64,'zero-head exact initial baseline')
        matrices=coding['matrix_records']
        if arm in ('P','S'):
            check(len(matrices)==2 and all(m['row_sum_max_error']<=6e-5 and m['cross_vertical_mass']==0 for m in matrices),'both geometry directions/row sums/vertical support')
        else:check(matrices is None,'generic/uniform never runs matching')
        for k,v in r['gradient_norms'].items():nonzero[k]+=int(v>0)
        frames.append(r['frame_id']);losses.append(r['loss'])
    check(all(v>0 for v in nonzero.values()),'entire training lacks native task branch')
    if steps==6:check(empty>0,'native sanity did not exercise known emptyGT case')
    summary=dict(steps=steps,unique_returned_frames=len(set(frames)),empty_GT_frames=empty,nonzero_gradient_frames=nonzero,
      first100_loss_median=statistics.median(losses[:100]),last100_loss_median=statistics.median(losses[-100:]),
      max_preclip_gradient=max(r['preclip_grad_norm'] for r in rows))
    return rows,summary

def expected_names():
    names=set()
    for branch,indices in {'stereo_encoder':[1,3],'stereo_decoder':[0,2],'appearance_encoder':[1,3],'appearance_decoder':[0,2]}.items():
        for index in indices:
            names.update('backbone_3d.stereo_feature_link.'+branch+'.'+str(index)+'.'+field for field in ('weight','bias'))
    return names

def audit_optimizer(saved,names,steps):
    opt=saved['optimizer_state'];check(len(opt['param_groups'])==1,'fresh AdamW group scope');g=opt['param_groups'][0]
    check(g['lr']==.0001 and g['weight_decay']==.0001 and g['betas']==(.9,.999) and g['eps']==1e-8
          and not g['amsgrad'] and not g['maximize'],'actual AdamW settings')
    check(g['parameter_names']==names and len(g['params'])==len(set(g['params']))==len(names) and set(g['params'])==set(opt['state']),'actual parameter/optimizer mapping')
    for name,key in zip(names,g['params']):
        state=opt['state'][key];value=saved['model_state'][name]
        check(float(state['step'])==steps,'actual Adam update count: '+name)
        for moment in ('exp_avg','exp_avg_sq'):
            check(state[moment].shape==value.shape and state[moment].dtype==value.dtype and torch.isfinite(state[moment]).all(),'actual finite moment: '+name)
        check((state['exp_avg_sq']>=0).all() and (state['exp_avg_sq']>0).any(),'actual learning signal: '+name)

def main():
    p=argparse.ArgumentParser()
    for name in ('manifest','initialization','config','protocol','fold','source-manifest','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--engineering-sanity',action='store_true');args=p.parse_args()
    if args.output.exists() or args.output.with_suffix('.records').exists():p.error('preserve previous audit')
    run=json.loads(args.manifest.read_text());steps=6 if args.engineering_sanity else STEPS;arm=run['arm']
    check(run['state']=='finished' and arm in ('U','G','P','S'),'finished explicit arm required')
    check(run['evidence_type']==('engineering_only' if args.engineering_sanity else 'exploratory_epipolar_conditioned_JSCC_native3D_adaptation'),'evidence scope')
    for path,key in [(args.initialization,'initialization_sha256'),(args.config,'config_sha256'),(args.protocol,'protocol_sha256'),
                     (args.fold,'fold_sha256'),(args.source_manifest,'source_manifest_sha256')]:check(sha256(path)==run[key],'input identity: '+key)
    check(run['initialization_sha256']==PARENT_SHA,'sole fixed F8 joint parent')
    check(run['protocol_sha256']==PROTOCOL_SHA and run['config_sha256']==CONFIG_SHA,'unchanged prelocked protocol/config')
    source=json.loads(args.source_manifest.read_text());identities=run['source_identities']
    check(set(source)==set(identities)=={'project','liga','mmdet','stereo_rcnn','experiment','F7','F8'},'all seven source trees')
    check(all(source[k]['actual_sources']==v for k,v in identities.items()),'source input differs')
    roots={'project':(ROOT,['src','scripts','configs','pyproject.toml']),
      'liga':(ROOT/'third_party/LIGA-Stereo',['liga','configs','tools','setup.py']),
      'mmdet':(ROOT/'third_party/mmdetection_kitti',['mmdet']),
      'stereo_rcnn':(ROOT/'third_party/Stereo-RCNN',['lib','demo.py','test_net.py']),
      'F7':(ROOT,['experiments/geometry-link/F7/code']),
      'F8':(ROOT,['experiments/geometry-link/F8/code']),
      'experiment':(ROOT,['experiments/geometry-link/F9/code'])}
    check(all(source_identity(*v)==identities[k] for k,v in roots.items()),'current frozen sources differ')
    original=json.loads((ROOT/'data/provenance/cao2025-staged-trainer-source-bfee9d8.json').read_text())
    check(run['original21_sources']==original and len(original)==21 and all(sha256(ROOT/'reproduction/cao2025'/k)==v for k,v in original.items()),'original21 unchanged')
    check(run['schedule']==schedule_manifest(),'exact predetermined SNR sequence')
    check(run['distributed_backend']=='nccl' and run['seed']==17,'native loss/RNG scope')
    check(run['budget']==dict(epochs=1,train_frames=STEPS,updates=STEPS,holdout_frames=372,batch_size=1,workers=4,
      lr=.0001,weight_decay=.0001,clip_norm=10,channel='awgn',SNR_uniform_dB=[0.,20.],SNR_seed=1927,noise_seed=1928,
      loss='native3D_classification_box_direction_no_teacher_depth_2D'),'fixed arm budget')
    check(run['optimizer_steps']==steps and run['completed_epochs']==int(not args.engineering_sanity),'actual budget/epochs')
    check(run['module_calls']==dict(steps=steps,student=2*steps,codec=steps,channel=steps,build_cost=steps,map_to_bev=steps,BEV=steps,head3D=steps,forbidden=0),'actual native module counts')
    check(set(run['training_inputs'])=={'batch_size','left_img','right_img','calib','image_shape','frame_id','random_T'}
      and run['target_input']=='augmented_gt_boxes_at_native3D_head_only' and run['all_modules_eval']
      and not run['original_teacher_calls_allowed'] and run['native_task_forward_allowed'],'GT/teacher/input scope')
    check(run['peak_reserved_GiB']*2**30+2*2**30<=run['gpu_free_before_bytes'],'measured physical margin')
    if args.engineering_sanity:
        profile=run['sender_profile']
        check(profile['scope']=='first_actual_step_from_sender_backbone_entry_to_channel_input' and profile['wall_seconds']>0 and profile['cuda_elapsed_ms']>=0,'passive actual sender timing scope')
        check(profile['new_head_parameters']==561 and profile['new_head_executed']==(arm!='U') and profile['correspondence_executed']==(arm in ('P','S')),'profile actual treatment compute')
        check(profile['operator_FLOPs_lower_bound']>0 and profile['recorded_sender_events']>0,'actual supported operator compute evidence')
        check(sha256(profile['profiler_trace_path'])==profile['profiler_trace_sha256'],'actual first-step profiler trace')
    checkpoint=Path(run['checkpoint_path']);check(sha256(checkpoint)==run['checkpoint_sha256'],'final checkpoint identity')
    actual=torch.load(checkpoint,map_location='cpu',weights_only=False)
    initial_path=Path(run['full_initialization_path']);check(sha256(initial_path)==run['full_initialization_sha256'],'complete initial snapshot identity')
    reference=torch.load(initial_path,map_location='cpu',weights_only=False)['model_state'];parent=torch.load(args.initialization,map_location='cpu',weights_only=False)['model_state']
    head_prefix='backbone_3d.stereo_feature_link.gain_head.'
    head_names={head_prefix+n for n in ('0.weight','0.bias','2.weight','2.bias')}
    check(len(parent)==535 and len(reference)==len(actual['model_state'])==539 and set(reference)==set(parent)|head_names,'535 parent plus four head states')
    check(all(v.dtype==reference[k].dtype and v.shape==reference[k].shape and torch.equal(v,reference[k]) for k,v in parent.items()),'whole exact parent values')
    # Independently construct the public initialization, without importing coupling.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1930)
        initial_head=torch.nn.Sequential(torch.nn.Linear(33,16),torch.nn.GELU(),torch.nn.Linear(16,1))
        torch.nn.init.zeros_(initial_head[2].weight);torch.nn.init.zeros_(initial_head[2].bias)
    check(sum(p.numel() for p in initial_head.parameters())==561,'head capacity')
    check(all(torch.equal(reference[head_prefix+n],v) for n,v in initial_head.state_dict().items()),'exact public isolated head initialization')
    names=run['trainable_parameter_names'];expected=expected_names()
    expected|={key for key in parent if key.startswith('backbone_3d.student_semantic_link_encoder.')}
    check(len(expected)==51,'existing35 student plus16 codec')
    if arm!='U':expected|=head_names
    count=51 if arm=='U' else 55;fixed=539-count
    check(len(names)==count and set(names)==expected,'declared codec/student scope')
    unchanged,changed=inactive_states(reference,actual['model_state'],expected)
    check(unchanged==run['frozen_states_identical']==fixed and set(changed)==expected,'exact fixed and changed scopes')
    check(actual['epoch']==(0 if args.engineering_sanity else 1) and actual['it']==steps and actual['arm']==arm,'saved counters/arm')
    check(actual['schedule']==schedule_manifest(),'saved channel schedule')
    audit_optimizer(actual,names,steps)
    fold=json.loads(args.fold.read_text());train=fold['folds']['geocomm_tune_train'];data=run['dataset'];root=Path(data['root'])
    check(data['split']=='geocomm_tune_train' and data['count']==len(train['ids'])==STEPS,'native full split')
    check((root/'ImageSets/geocomm_tune_train.txt').read_text().split()==train['ids'] and data['split_sha256']==train['split_sha256']==sha256(root/'ImageSets/geocomm_tune_train.txt'),'native split identity')
    check(data['infos']['kitti_infos_geocomm_tune_train.pkl']==train['infos_sha256']==sha256(root/'kitti_infos_geocomm_tune_train.pkl'),'native infos')
    check(data['download_manifest_sha256']==sha256(root/'download-manifest.json') and not set(train['ids'])&set(fold['folds']['geocomm_tune_holdout']['ids']),'archive/split separation')
    records=Path(run['output_dir'])/'training.jsonl';check(sha256(records)==run['training_records_sha256'],'raw records identity')
    config=yaml.safe_load(args.config.read_text());iou=config['MODEL']['DENSE_HEAD']['LOSS_CONFIG']['LOSS_WEIGHTS']['iou_weight']
    rows,summary=scalar_records(records,set(train['ids']),steps,iou,arm);check(summary['unique_returned_frames']==run['unique_returned_training_frames'],'returned frame count')
    replay=replay_rng_checkpoint(actual['channel_rng'],'awgn',steps,rows)
    for row in rows:
        check(row['optimization_scope']==arm and set(row['parameter_gradients'])==expected,'recorded selected gradient scope')
        check(all(g['finite'] and math.isfinite(g['norm']) and g['norm']>=0 and type(g['nonzero']) is int and g['nonzero']>=0 for g in row['parameter_gradients'].values()),'actual finite selected gradients')
    check(all(any(row['parameter_gradients'][name]['nonzero']>0 for row in rows) for name in expected),'every selected tensor received learning signal')
    snapshot=args.output.with_suffix('.records');snapshot.mkdir(parents=True,exist_ok=False);shutil.copyfile(records,snapshot/'training.jsonl')
    check(sha256(snapshot/'training.jsonl')==sha256(records),'sealed raw snapshot')
    result=dict(state='passed',arm=arm,evidence_type=run['evidence_type'],steps=steps,inactive_states_identical=fixed,trained_parameter_tensors=count,
      all_optimizer_update_counts=steps,summary=summary,noise_rng_replay=replay,checkpoint_sha256=run['checkpoint_sha256'],
      full_initialization_sha256=run['full_initialization_sha256'],manifest_sha256=sha256(args.manifest),source_manifest_sha256=sha256(args.source_manifest),
      raw_snapshot_sha256=sha256(snapshot/'training.jsonl'),limitations='Independent full saved-state/actualAdam/scalar/schedule/noise-draw replay audit; no fresh native loss forward or AP proof')
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
