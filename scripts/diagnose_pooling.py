#!/usr/bin/env python3
"""Locked native clean-feature pooling suite; no training or wireless channel."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from geocomm.pooling_diagnostic import CONDITIONS,PoolingDiagnostic,state_hashes
from geocomm.evidence import sha256,source_identity

CHECKPOINT_SHA='984771355f145441f785a8bca5ddde2762fec4925d504298b7fb9abe8c3f57f0'


def unused_link_states():
    names=set()
    branches={'cost_encoder':(1,3),'cost_decoder':(0,2),'appearance_encoder':(1,3),
              'appearance_decoder':(0,2),'posterior':(0,2),
              'sensitivity_predictor.cost':(0,2),'sensitivity_predictor.appearance':(0,2)}
    for branch,indices in branches.items():
        for index in indices:
            for field in ('weight','bias'): names.add('backbone_3d.semantic_link.'+branch+'.'+str(index)+'.'+field)
    names.update('backbone_3d.semantic_link.generic_score.'+field for field in ('weight','bias'))
    return names


def frozen_load(model,path,loader=None,kwargs=None):
    if sha256(path)!=CHECKPOINT_SHA: raise ValueError('exact fixed F4 checkpoint required')
    from geocomm.compat import adapt_spconv_state
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    state=adapt_spconv_state(model,checkpoint['model_state']);expected=model.state_dict()
    if len(expected)!=519 or len(state)!=549 or set(state)-set(expected)!=unused_link_states() or not set(expected)<=set(state):
        raise ValueError('complete519 required states and exact30 unused link states differ')
    for name,value in expected.items():
        if value.shape!=state[name].shape or value.dtype!=state[name].dtype or not torch.isfinite(state[name]).all():
            raise ValueError('required state layout/finite differs: '+name)
    if loader is None: model.load_state_dict({name:state[name] for name in expected},strict=True)
    else: loader(filename=str(path),**(kwargs or {}))
    if any(not torch.equal(value.detach().cpu(),state[name]) for name,value in model.state_dict().items()):
        raise RuntimeError('actual native loader did not preserve complete required checkpoint values')
    # Keep native gradient flags for the upstream DDP evaluator. No optimizer
    # exists; hooks require no_grad eval and every state is checked afterwards.
    if not any(p.requires_grad for p in model.parameters()):
        raise RuntimeError('native DDP requires at least one enabled parameter flag')
    return dict(required_states=519,unused_link_states=sorted(unused_link_states()),state_hashes=state_hashes(model))


def main():
    p=argparse.ArgumentParser();p.add_argument('--condition',choices=CONDITIONS,required=True)
    p.add_argument('--records',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    args,remainder=p.parse_known_args()
    if not remainder or remainder[0]!='test' or args.records.exists() or args.report.exists():
        p.error('test-only arguments and unique evidence outputs required')
    args.records=args.records.resolve();args.report=args.report.resolve()
    args.records.parent.mkdir(parents=True,exist_ok=True);args.report.parent.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=[str(ROOT/'third_party/LIGA-Stereo'),str(ROOT/'third_party/LIGA-Stereo/tools'),str(ROOT/'third_party/mmdetection_kitti')]
    import liga.models
    original=liga.models.build_network;models=[];observers=[];loads=[];frames=[]
    report=dict(state='starting',scope='fixed clean-feature pooling diagnosis; no communication/training',
        condition=args.condition,started_at_unix=time.time(),project_sources=source_identity(ROOT,['src','scripts','configs','pyproject.toml']))
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    with args.records.open('x') as stream:
        def record(row):
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();frames.append(row['frame_id'])
        def builder(*pargs,**kwargs):
            if models: raise RuntimeError('one native model required')
            model=original(*pargs,**kwargs);models.append(model)
            native_loader=model.load_params_from_file
            def checked_loader(filename,**load_kwargs):
                if loads: raise RuntimeError('exactly one fixed checkpoint load required')
                loads.append(frozen_load(model,Path(filename),native_loader,load_kwargs))
            model.load_params_from_file=checked_loader
            observers.append(PoolingDiagnostic(model,args.condition,record));return model
        liga.models.build_network=builder
        try:
            sys.argv=[str(ROOT/'scripts/run_liga.py')]+remainder
            runpy.run_path(str(ROOT/'scripts/run_liga.py'),run_name='__main__')
            fold=json.loads((ROOT/'data/internal-tuning-fold-001.json').read_text())
            if frames!=fold['folds']['geocomm_tune_holdout']['ids']: raise RuntimeError('all ordered372 diagnostic frames required')
            if len(loads)!=1 or state_hashes(models[0])!=loads[0]['state_hashes']: raise RuntimeError('required model state changed during inference')
            if observers[0].calls!=dict(frames=372,student=744,raw_cost=372,forbidden=0): raise RuntimeError('native module counts differ')
            report.update(state='finished',frames=372,calls=observers[0].calls,checkpoint_sha256=CHECKPOINT_SHA,
                checkpoint_loading=loads[0],final_state_hashes=state_hashes(models[0]),records_sha256=sha256(args.records))
        except BaseException as error:
            report.update(state='failed',exception=repr(error));raise
        finally:
            for observer in observers: observer.close()
            liga.models.build_network=original
            report['ended_at_unix']=time.time();args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__': main()
