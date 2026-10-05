"""Complete full-native five-stage trainer for the explicit original-paper variant."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import StereoRGB,digest,single_frame
from model import SemanticVariant
from wireless import WirelessVariant
from stages import EPOCHS,configure,forward_loss,frozen_state_keys,optimizer_groups,phase

ROOT=Path(__file__).resolve().parent


def sources():
    paths=list(ROOT.rglob('*.py'))+list((ROOT/'upstream').glob('*.json'))
    paths += [ROOT/'upstream/LICENSE.txt']
    return {str(path.relative_to(ROOT)):digest(path) for path in sorted(paths) if '__pycache__' not in path.parts}


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def state_hash(value):
    raw=value.detach().cpu().contiguous()
    return hashlib.sha256(str(raw.dtype).encode()+str(tuple(raw.shape)).encode()+raw.numpy().tobytes()).hexdigest()


def evaluate(model,loader,current,output):
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
    rows=0;erasures=0;loss_sum=0.
    before={name:state_hash(value) for name,value in model.state_dict().items()}
    model.eval();device=next(model.parameters()).device;frame_ids=[]
    try:
        with torch.no_grad(),output.open('x') as stream:
            for batch in loader:
                result=forward_loss(model,batch['left'].to(device,non_blocking=True),batch['right'].to(device,non_blocking=True),
                                    batch['boxes'],current,snr_db=10.)
                loss=float(result['loss']) if result['loss'] is not None else None
                if loss is not None and not np.isfinite(loss): raise RuntimeError('nonfinite final-stage holdout loss')
                stream.write(json.dumps(dict(frame_id=batch['frame_id'],loss=loss,
                    erasure=result['erasure'],accounting=result['accounting']),allow_nan=False)+'\n')
                rows+=1;frame_ids.append(batch['frame_id']);erasures+=int(result['erasure'] is not None);loss_sum+=loss or 0.
        if rows!=372: raise RuntimeError('incomplete final-stage holdout coverage')
        if frame_ids!=[row['frame_id'] for row in loader.dataset.rows]: raise RuntimeError('holdout order/IDs differ')
        if any(state_hash(value)!=before[name] for name,value in model.state_dict().items()):
            raise RuntimeError('evaluation mutated saved final-stage model state')
        return dict(frames=rows,erasures=erasures,successful_frames=rows-erasures,
                    mean_loss_on_successes=loss_sum/(rows-erasures) if rows>erasures else None,records_sha256=digest(output))
    finally:
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',type=int,choices=list(EPOCHS),required=True)
    for name in ('spynet','data-root','train-records','train-audit','holdout-records','holdout-audit','fold','protocol','output-dir','manifest'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--predecessor-manifest',type=Path)
    parser.add_argument('--predecessor-audit',type=Path)
    parser.add_argument('--engineering-steps',type=int,default=0)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args=parser.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path): setattr(args,key,value.resolve())
    if args.output_dir.exists() or args.manifest.exists() or args.engineering_steps<0:
        parser.error('unique outputs and nonnegative engineering steps required')
    if not args.engineering_steps and args.device!='cuda': parser.error('formal native GPU training required')
    if args.engineering_steps>3340: parser.error('engineering must fit in one partial epoch')
    checkpoint_metadata=json.loads((ROOT/'upstream/checkpoint.json').read_text())
    if digest(args.spynet)!=checkpoint_metadata['sha256']: parser.error('fixed official SpyNet release required')
    if args.stage>1 and (args.predecessor_manifest is None or args.predecessor_audit is None):
        parser.error('complete audited predecessor required')
    args.output_dir.mkdir(parents=True);args.manifest.parent.mkdir(parents=True,exist_ok=True)
    identity=sources()
    info=dict(state='starting',evidence_type='engineering_only' if args.engineering_steps else 'exploratory_full_native_original_variant',
        stage=args.stage,seed=17,device=args.device,started_at_unix=time.time(),output_dir=str(args.output_dir),
        source_files=identity,protocol_sha256=digest(args.protocol),fold_sha256=digest(args.fold),spynet_sha256=digest(args.spynet),
        train_records_sha256=digest(args.train_records),train_audit_sha256=digest(args.train_audit),
        holdout_records_sha256=digest(args.holdout_records),holdout_audit_sha256=digest(args.holdout_audit),
        torch_version=torch.__version__,epochs=[],optimizer_steps=0,samples=0,
        budget=dict(epochs=EPOCHS[args.stage],train_frames=3340,holdout_frames=372,batch_size=1,workers=4,
            optimizer='Adam',weight_decay=0.,clip_norm=10.,charbonnier_epsilon=.001,
            stage4_hybrid_epochs=5,stage5_channel_epochs=25,stage5_train_snr=[6.,18.],native_rgb=True,augmentation='none'))
    write_json(args.manifest,info)
    try:
        torch.set_num_threads(4);random.seed(17);np.random.seed(17);torch.manual_seed(17)
        if args.device=='cuda': torch.cuda.manual_seed_all(17)
        torch.backends.cudnn.benchmark=False
        train=StereoRGB(args.data_root,args.train_records,args.train_audit,args.fold,'geocomm_tune_train')
        holdout=StereoRGB(args.data_root,args.holdout_records,args.holdout_audit,args.fold,'geocomm_tune_holdout')
        if len(train)!=3340 or len(holdout)!=372: raise RuntimeError('native full-fold budget differs')
        generator=torch.Generator().manual_seed(17+1000*args.stage)
        loader=DataLoader(train,batch_size=1,shuffle=True,num_workers=4,pin_memory=args.device=='cuda',
            collate_fn=single_frame,generator=generator,multiprocessing_context='spawn')
        validation=DataLoader(holdout,batch_size=1,shuffle=False,num_workers=4,pin_memory=True,
            collate_fn=single_frame,multiprocessing_context='spawn')
        model=WirelessVariant(SemanticVariant(args.spynet)).to(args.device)
        if args.stage>1:
            parent=json.loads(args.predecessor_manifest.read_text());audit=json.loads(args.predecessor_audit.read_text())
            if (parent['state']!='finished' or parent['stage']!=args.stage-1 or audit['state']!='passed'
                    or parent['evidence_type']!='exploratory_full_native_original_variant'
                    or audit['manifest_sha256']!=digest(args.predecessor_manifest)):
                raise RuntimeError('predecessor scope/completion/audit differs')
            parent_checkpoint=Path(parent['epochs'][-1]['checkpoint_path'])
            if digest(parent_checkpoint)!=parent['epochs'][-1]['checkpoint_sha256']: raise RuntimeError('predecessor checkpoint identity differs')
            previous=torch.load(parent_checkpoint,map_location='cpu',weights_only=False)
            model.load_state_dict(previous['model_state'],strict=True)
            info.update(predecessor_manifest_sha256=digest(args.predecessor_manifest),
                predecessor_audit_sha256=digest(args.predecessor_audit),predecessor_checkpoint_sha256=digest(parent_checkpoint))
            del previous
        if sum(p.numel() for p in model.parameters())!=12970798 or len(model.state_dict())!=768:
            raise RuntimeError('complete documented architecture differs')
        initial_path=args.output_dir/'initialization.pth'
        torch.save(dict(model_state={name:value.detach().cpu() for name,value in model.state_dict().items()}),initial_path)
        info.update(initialization_path=str(initial_path),initialization_sha256=digest(initial_path),
                    parameter_tensors=sum(1 for _ in model.parameters()),state_tensors=768,
                    state='training',shuffle_generator_seed=17+1000*args.stage)
        groups=optimizer_groups(model,args.stage);optimizer=torch.optim.Adam(groups,weight_decay=0.)
        parameters=dict(model.named_parameters());counts={name:0 for name in parameters}
        attempts=updates=0;write_json(args.manifest,info)
        raw_path=args.output_dir/'training.jsonl'
        with raw_path.open('x') as stream:
            for epoch in range(1,EPOCHS[args.stage]+1):
                if sources()!=identity or digest(args.protocol)!=info['protocol_sha256']:
                    raise RuntimeError('original reproduction sources/protocol changed before epoch')
                current=phase(args.stage,epoch);active=configure(model,current)
                frozen=frozen_state_keys(model)
                before={name:state_hash(model.state_dict()[name]) for name in frozen}
                epoch_ids=[];epoch_erasures=0;epoch_updates=0
                for batch in loader:
                    model.zero_grad(set_to_none=True)
                    left=batch['left'].to(args.device,non_blocking=True);right=batch['right'].to(args.device,non_blocking=True)
                    snr=float(torch.empty(()).uniform_(6,18)) if args.stage==5 else None
                    result=forward_loss(model,left,right,batch['boxes'],current,snr_db=snr or 10.)
                    attempts+=1;epoch_ids.append(batch['frame_id']);grad_names=[];norm=None;loss=None
                    if result['erasure'] is None:
                        loss_tensor=result['loss']
                        if not torch.isfinite(loss_tensor): raise RuntimeError('nonfinite loss before original Adam update')
                        loss_tensor.backward();grad_names=[name for name,p in parameters.items() if p.grad is not None]
                        if not set(grad_names)<=active: raise RuntimeError('frozen parameter acquired gradient')
                        expected=active.copy()
                        if args.stage==5 and epoch<=25 and sum(result['accounting']['key_cells'])==0:
                            expected={name for name in expected if not name.startswith('key_channel.decoder.')}
                        if set(grad_names)!=expected: raise RuntimeError('active gradient coverage differs')
                        norm=float(torch.nn.utils.clip_grad_norm_([parameters[name] for name in grad_names],10.,error_if_nonfinite=True))
                        optimizer.step();updates+=1;epoch_updates+=1
                        for name in grad_names: counts[name]+=1
                        loss=float(loss_tensor.detach())
                    else: epoch_erasures+=1
                    record=dict(sample=attempts,optimizer_updates=updates,epoch=epoch,frame_id=batch['frame_id'],
                        shape=list(left.shape),box_counts=[len(v) for v in batch['boxes']],loss_type=current['loss'],
                        loss=loss,preclip_grad_norm=norm,updated=result['erasure'] is None,
                        gradient_parameter_tensors=len(grad_names),erasure=result['erasure'],snr_db=snr,accounting=result['accounting'])
                    stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
                    info.update(samples=attempts,optimizer_steps=updates)
                    if attempts==1 or attempts%100==0:
                        write_json(args.manifest,info);print(json.dumps({k:record[k] for k in ['sample','optimizer_updates','epoch','loss','erasure']}),flush=True)
                    if args.engineering_steps and attempts>=args.engineering_steps: break
                if not args.engineering_steps and (len(epoch_ids)!=3340 or set(epoch_ids)!={r['frame_id'] for r in train.rows}):
                    raise RuntimeError('native RGB epoch must cover each scheduled frame exactly once')
                if any(state_hash(model.state_dict()[name])!=value for name,value in before.items()):
                    raise RuntimeError('inactive original state changed during epoch')
                if sources()!=identity or digest(args.protocol)!=info['protocol_sha256']:
                    raise RuntimeError('original sources/protocol changed during epoch')
                snapshot=args.output_dir/f'checkpoint_epoch_{epoch if not args.engineering_steps else 0}.pth'
                torch.save(dict(model_state={name:value.detach().cpu() for name,value in model.state_dict().items()},
                    optimizer_state=optimizer.state_dict(),stage=args.stage,epoch=epoch if not args.engineering_steps else 0,
                    samples=attempts,optimizer_updates=updates,parameter_update_counts=counts.copy(),
                    rng_state=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                        cuda=torch.cuda.get_rng_state_all() if args.device=='cuda' else [],shuffle=generator.get_state())),snapshot)
                info['epochs'].append(dict(epoch=epoch,phase=current,samples=len(epoch_ids),updates=epoch_updates,
                    erasures=epoch_erasures,frozen_state_tensors=len(frozen),frozen_states_identical=True,
                    checkpoint_path=str(snapshot),checkpoint_sha256=digest(snapshot),parameter_update_counts=counts.copy()))
                write_json(args.manifest,info)
                if args.engineering_steps: break
        if not args.engineering_steps:
            if attempts!=3340*EPOCHS[args.stage]: raise RuntimeError('complete stage sample budget differs')
            info['final_holdout']=evaluate(model,validation,phase(args.stage,EPOCHS[args.stage]),args.output_dir/'final_holdout.jsonl')
        if sources()!=identity or digest(args.protocol)!=info['protocol_sha256']:
            raise RuntimeError('original source/protocol changed before completion')
        info.update(state='finished',ended_at_unix=time.time(),training_records_sha256=digest(raw_path))
        write_json(args.manifest,info);print(json.dumps(dict(state=info['state'],stage=args.stage,samples=attempts,updates=updates)))
    except BaseException as error:
        info.update(state='failed',exception=repr(error),ended_at_unix=time.time());write_json(args.manifest,info);raise


if __name__=='__main__': main()
