"""Independent complete saved-state/counter/scalar audit of original stages.

Schedule and transport formula are independently expressed, not imported from
training/stages/radio. Architecture is used only to enumerate canonical states.
No fresh inference, loss recomputation or AP claim follows from this audit.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import numpy as np
import torch
from model import SemanticVariant
from wireless import WirelessVariant

ROOT=Path(__file__).resolve().parent
EPOCHS={1:12,2:10,3:6,4:10,5:45}
COMPONENTS=('global_encoder','global_decoder','flow','key_encoders','key_decoders','fusions','global_channel','key_channel')


def check(condition,message):
    if not condition: raise ValueError(message)


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def component(name):
    if name.startswith('semantic.global_decoder.flow.'): return 'flow'
    for value in COMPONENTS:
        if name.startswith(('' if value.endswith('_channel') else 'semantic.')+value+'.'): return value
    raise ValueError('unmapped complete state: '+name)


def policy(stage,epoch):
    if stage==1:
        rates={'global_encoder':2e-4,'global_decoder':2e-4,'fusions':1e-4}
        if epoch>2: rates['flow']=2.5e-5
        loss='global_charbonnier'
    elif stage==2: rates={'key_encoders':1e-4,'key_decoders':2e-4};loss='masked_charbonnier'
    elif stage==3: rates={'fusions':1e-4};loss='hybrid_charbonnier'
    else:
        rates={'global_encoder':1e-4,'global_decoder':1e-4,'key_encoders':2e-5,'key_decoders':2e-5,'fusions':2e-5,'flow':1.25e-5}
        if stage==5:
            rates.update(global_channel=1e-4,key_channel=1e-4)
            if epoch<=25: rates={k:rates[k] for k in ('global_channel','key_channel')}
            loss='semantic_mse' if epoch<=25 else 'global_charbonnier'
        else: loss='hybrid_charbonnier' if epoch<=5 else 'global_charbonnier'
    return rates,loss


def layout(row):
    h,w=row['views'][0]['shape'];hp,wp=h+(-h)%6,w+(-w)%6;cells=[]
    for view in row['views']:
        occupied=np.zeros((hp//2,wp//2),dtype=bool)
        for box in view['boxes']:
            x1,y1,x2,y2=box['xyxy'];occupied[y1//2:(y2+1)//2,x1//2:(x2+1)//2]=True
        cells.append(int(occupied.sum()))
    real=9*(2*(hp//6)*(wp//6)+sum(cells));data=(real+1)//2
    control=1176+672*sum(len(v['boxes']) for v in row['views'])
    return dict(key_cells=cells,data_real_values=real,padding_real_values=real%2,
                data_uses=data,control_uses=control,pilot_uses=0,total_uses=data+control,
                cbr_complex_per_rgb_real_value=(data+control)/(6*h*w))


def accounting(value,row):
    expected=layout(row)
    check(set(value)==set(expected)|{'total_energy'},'wire accounting fields differ')
    for key,wanted in expected.items():
        if key.startswith('cbr'): check(math.isclose(value[key],wanted,rel_tol=1e-12),'wire CBR differs')
        else: check(value[key]==wanted,'wire layout differs: '+key)
    check(math.isfinite(value['total_energy']) and abs(value['total_energy']/value['total_uses']-1)<=1e-4,'unit physical energy differs')
    return expected


def validate_states(initial,saved,active,stage):
    check(set(initial)==set(saved) and len(saved)==768,'full768-state coverage differs')
    frozen=0;changed=0
    for name,value in saved.items():
        reference=initial[name]
        check(value.dtype==reference.dtype and value.shape==reference.shape and torch.isfinite(value).all(),'state layout/finite differs: '+name)
        allowed=name in active or (stage==5 and name.startswith(('global_channel.','key_channel.')))
        if not allowed:
            check(torch.equal(reference,value),'inactive state changed: '+name);frozen+=1
        else: changed+=int(not torch.equal(reference,value))
    check(frozen==768-len(active)-(48 if stage==5 else 0),'inactive state count differs')
    return frozen,changed


def optimizer_audit(saved,names,counts,stage):
    opt=saved['optimizer_state'];rates,_=policy(stage,EPOCHS[stage]);mapping={};seen=[]
    check(len(opt['param_groups'])==len(rates),'optimizer groups differ')
    for group in opt['param_groups']:
        key=group['component'];check(key in rates and key not in seen,'optimizer component differs');seen.append(key)
        expected=[name for name in names if component(name)==key]
        check(group['parameter_names']==expected and len(group['params'])==len(expected),'canonical optimizer mapping differs')
        check(group['lr']==rates[key] and group['weight_decay']==0. and group['betas']==(.9,.999) and group['eps']==1e-8 and not group['amsgrad'],'Adam settings differ')
        for index,name in zip(group['params'],expected):
            check(index not in mapping,'duplicate optimizer parameter');mapping[index]=name
    check(set(opt['state'])=={index for index,name in mapping.items() if counts[name]>0},'actual optimizer state/update coverage differs')
    for index,value in opt['state'].items():
        name=mapping[index];check(float(value['step'])==counts[name],'actual per-parameter Adam steps differ: '+name)
        for key in ('exp_avg','exp_avg_sq'):
            check(value[key].shape==saved['model_state'][name].shape and torch.isfinite(value[key]).all(),'moment invalid: '+name)
        check((value['exp_avg_sq']>=0).all(),'negative second moment')
    return len(opt['state'])


def main():
    parser=argparse.ArgumentParser()
    for name in ('manifest','spynet','train-records','train-audit','holdout-records','holdout-audit','fold','protocol','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--predecessor-manifest',type=Path);parser.add_argument('--predecessor-audit',type=Path)
    parser.add_argument('--engineering',action='store_true');args=parser.parse_args()
    check(not args.output.exists() and not args.output.with_suffix('.records').exists(),'preserve previous audit evidence')
    run=json.loads(args.manifest.read_text());stage=run['stage'];check(stage in EPOCHS and run['state']=='finished','stage unfinished')
    check(run['evidence_type']==('engineering_only' if args.engineering else 'exploratory_full_native_original_variant'),'evidence scope differs')
    check(run['seed']==17 and run['shuffle_generator_seed']==17+1000*stage,'seed differs')
    check(run['parameter_tensors']==718 and run['state_tensors']==768,'complete architecture coverage differs')
    for key in ('spynet','train-records','train-audit','holdout-records','holdout-audit','fold','protocol'):
        check(digest(getattr(args,key.replace('-','_')))==run[key.replace('-','_')+'_sha256'],'input identity differs: '+key)
    budget=dict(epochs=EPOCHS[stage],train_frames=3340,holdout_frames=372,batch_size=1,workers=4,
        optimizer='Adam',weight_decay=0.,clip_norm=10.,charbonnier_epsilon=.001,
        stage4_hybrid_epochs=5,stage5_channel_epochs=25,stage5_train_snr=[6.,18.],native_rgb=True,augmentation='none')
    check(run['budget']==budget and (run['device']=='cuda' or args.engineering),'formal native GPU/budget differs')
    current_sources={str(p.relative_to(ROOT)):digest(p) for p in sorted(list(ROOT.rglob('*.py'))+list((ROOT/'upstream').glob('*.json'))+[ROOT/'upstream/LICENSE.txt']) if '__pycache__' not in p.parts}
    check(current_sources==run['source_files'],'source identity changed')
    fold=json.loads(args.fold.read_text());rows={};ids={}
    for split,records,audit in [('geocomm_tune_train',args.train_records,args.train_audit),('geocomm_tune_holdout',args.holdout_records,args.holdout_audit)]:
        data=[json.loads(line) for line in records.read_text().splitlines()];a=json.loads(audit.read_text())
        ids[split]=fold['folds'][split]['ids'];check([row['frame_id'] for row in data]==ids[split],'fixed ROI cache coverage/order differs')
        check(a['state']=='passed' and a['records_sha256']==digest(records) and a['fold_sha256']==digest(args.fold) and a['split']==split and a['frames']==len(data),'ROI audit identity differs')
        rows[split]={row['frame_id']:row for row in data}
    check(len(ids['geocomm_tune_train'])==3340 and len(ids['geocomm_tune_holdout'])==372 and not set(ids['geocomm_tune_train'])&set(ids['geocomm_tune_holdout']),'native fold counts/overlap differs')
    initial_path=Path(run['initialization_path']);check(digest(initial_path)==run['initialization_sha256'],'initialization identity differs')
    initial=torch.load(initial_path,map_location='cpu',weights_only=False)['model_state']
    torch.set_num_threads(4);torch.manual_seed(17);model=WirelessVariant(SemanticVariant(args.spynet));names=list(dict(model.named_parameters()))
    check(len(names)==718 and sum(p.numel() for p in model.parameters())==12970798,'canonical parameter coverage differs')
    if stage==1:
        check(all(torch.equal(initial[name],value) for name,value in model.state_dict().items()),'fresh seed17 model/strict SpyNet initialization differs')
    else:
        check(args.predecessor_manifest is not None and args.predecessor_audit is not None,'audited predecessor required')
        parent=json.loads(args.predecessor_manifest.read_text());a=json.loads(args.predecessor_audit.read_text())
        check(parent['state']=='finished' and parent['stage']==stage-1 and parent['evidence_type']=='exploratory_full_native_original_variant' and a['state']=='passed' and a['manifest_sha256']==digest(args.predecessor_manifest),'predecessor scope/audit differs')
        for path,key in [(args.predecessor_manifest,'predecessor_manifest_sha256'),(args.predecessor_audit,'predecessor_audit_sha256')]: check(digest(path)==run[key],'predecessor input identity differs')
        path=Path(parent['epochs'][-1]['checkpoint_path']);check(digest(path)==parent['epochs'][-1]['checkpoint_sha256']==run['predecessor_checkpoint_sha256'],'predecessor checkpoint identity differs')
        previous=torch.load(path,map_location='cpu',weights_only=False)['model_state'];check(set(previous)==set(initial) and all(torch.equal(previous[k],initial[k]) for k in initial),'whole predecessor initialization differs');del previous
    del model
    raw=Path(run['output_dir'])/'training.jsonl';check(digest(raw)==run['training_records_sha256'],'raw training identity differs')
    snapshots=args.output.with_suffix('.records');snapshots.mkdir(parents=True);shutil.copyfile(raw,snapshots/'training.jsonl')
    records=[json.loads(line) for line in raw.read_text().splitlines()];check(len(records)==run['samples'],'sample count differs')
    check(len(run['epochs'])==(1 if args.engineering else EPOCHS[stage]),'completed epoch count differs')
    check((0<len(records)<=3340) if args.engineering else len(records)==3340*EPOCHS[stage],'complete sample budget differs')
    counts={name:0 for name in names};updates=attempts=successful=0;epoch_audits=[]
    for epoch_info in run['epochs']:
        epoch=epoch_info['epoch'];rates,loss_type=policy(stage,epoch);active={name for name in names if component(name) in rates}
        check(epoch_info['phase']==dict(stage=stage,epoch=epoch,rates=rates,loss=loss_type),'phase schedule differs')
        chunk=records[(epoch-1)*3340:epoch*3340];visited=[];erased=0;epoch_updates=0
        for row in chunk:
            attempts+=1;frame=row['frame_id'];check(frame in rows['geocomm_tune_train'],'frame outside scheduled fold');native=rows['geocomm_tune_train'][frame];visited.append(frame)
            check(row['sample']==attempts and row['epoch']==epoch and row['loss_type']==loss_type,'scalar counters/schedule differ')
            check(row['shape']==[1,3,*native['views'][0]['shape']] and row['box_counts']==[len(v['boxes']) for v in native['views']],'native frame geometry/ROI differs')
            expected=active.copy()
            if stage==5:
                wire=accounting(row['accounting'],native);check(math.isfinite(row['snr_db']) and 6<=row['snr_db']<=18,'training SNR differs')
                if epoch<=25 and sum(wire['key_cells'])==0: expected={name for name in active if not name.startswith('key_channel.decoder.')}
            else: check(row['accounting'] is None and row['snr_db'] is None and row['erasure'] is None,'nonwireless stage transport differs')
            if row['erasure'] is None:
                check(row['updated'] and row['gradient_parameter_tensors']==len(expected),'actual gradient/update scope differs')
                check(all(math.isfinite(row[key]) and row[key]>=0 for key in ('loss','preclip_grad_norm')),'invalid update scalar')
                updates+=1;epoch_updates+=1;successful+=1
                for name in expected: counts[name]+=1
            else:
                erased+=1;check(isinstance(row['erasure'],str) and not row['updated'] and row['loss'] is None and row['preclip_grad_norm'] is None and row['gradient_parameter_tensors']==0,'erasure/update differs')
            check(row['optimizer_updates']==updates,'contiguous successful update counter differs')
        check(len(visited)==len(set(visited)) and (args.engineering or set(visited)==set(ids['geocomm_tune_train'])),'epoch frame coverage differs')
        check(epoch_info['samples']==len(chunk) and epoch_info['updates']==epoch_updates and epoch_info['erasures']==erased,'epoch summary counters differ')
        path=Path(epoch_info['checkpoint_path']);check(digest(path)==epoch_info['checkpoint_sha256'],'checkpoint identity differs')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        check(saved['stage']==stage and saved['epoch']==(0 if args.engineering else epoch) and saved['samples']==attempts and saved['optimizer_updates']==updates,'checkpoint counters differ')
        check(saved['parameter_update_counts']==epoch_info['parameter_update_counts']==counts,'per-parameter update counters differ')
        frozen,changed=validate_states(initial,saved['model_state'],active,stage)
        check(epoch_info['frozen_states_identical'] and epoch_info['frozen_state_tensors']==frozen,'inactive state summary differs')
        opt_states=optimizer_audit(saved,names,counts,stage)
        for name,value in saved['model_state'].items():
            if name.endswith('num_batches_tracked'):
                increment=(2*attempts if '.encoder.' in name else 2*successful) if stage==5 else 0
                check(int(value)==int(initial[name])+increment,'BatchNorm frame/call counters differ: '+name)
        epoch_audits.append(dict(epoch=epoch,samples=len(chunk),updates=epoch_updates,erasures=erased,
            active_parameter_tensors=len(active),inactive_states_identical=frozen,changed_allowed_states=changed,optimizer_states=opt_states,checkpoint_sha256=digest(path)))
        del saved
    check(attempts==run['samples'] and updates==run['optimizer_steps'],'final stage counters differ')
    if not args.engineering:
        path=Path(run['output_dir'])/'final_holdout.jsonl';check(digest(path)==run['final_holdout']['records_sha256'],'holdout raw identity differs')
        shutil.copyfile(path,snapshots/'final_holdout.jsonl');values=[json.loads(line) for line in path.read_text().splitlines()]
        check([row['frame_id'] for row in values]==ids['geocomm_tune_holdout'],'complete fixed holdout order/IDs differ')
        losses=[];erased=0
        for row in values:
            if stage==5: accounting(row['accounting'],rows['geocomm_tune_holdout'][row['frame_id']])
            else: check(row['accounting'] is None and row['erasure'] is None,'holdout stage scope differs')
            if row['erasure'] is None:
                check(math.isfinite(row['loss']) and row['loss']>=0,'holdout loss invalid');losses.append(row['loss'])
            else: check(row['loss'] is None and isinstance(row['erasure'],str),'holdout erasure differs');erased+=1
        expected=dict(frames=372,erasures=erased,successful_frames=len(losses),mean_loss_on_successes=sum(losses)/len(losses) if losses else None,records_sha256=digest(path))
        check(expected==run['final_holdout'],'holdout scalar reduction differs')
    result=dict(state='passed',stage=stage,evidence_type=run['evidence_type'],manifest_sha256=digest(args.manifest),
        epochs=epoch_audits,samples=attempts,optimizer_updates=updates,checkpoint_sha256=epoch_audits[-1]['checkpoint_sha256'],
        source_files=run['source_files'],raw_snapshot_sha256=digest(snapshots/'training.jsonl'),
        limitations='independent complete saved-state/optimizer/scalar/layout audit; image identities inherited from audited ROI cache and loader; no fresh inference or AP proof')
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))


if __name__=='__main__': main()
