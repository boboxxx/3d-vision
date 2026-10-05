#!/usr/bin/env python3
"""Full native F5 evaluator: exact535 load, passive features, receiver cost boundary."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from geocomm.stereo_feature_observer import StereoFeatureObserver
from geocomm.pooling_diagnostic import state_hashes
from geocomm.evidence import sha256,source_identity
from diagnose_codec_features import compare_features


def frozen_load(model,path,expected_sha,loader=None,kwargs=None):
    if sha256(path)!=expected_sha: raise ValueError('fixed final F5 checkpoint identity differs')
    from geocomm.compat import adapt_spconv_state
    state=adapt_spconv_state(model,torch.load(path,map_location='cpu',weights_only=False)['model_state'])
    expected=model.state_dict()
    if len(expected)!=535 or set(state)!=set(expected): raise ValueError('complete535-state checkpoint required')
    for name,value in expected.items():
        if value.shape!=state[name].shape or value.dtype!=state[name].dtype or not torch.isfinite(state[name]).all():
            raise ValueError('state layout/finite differs: '+name)
    if loader is None: model.load_state_dict(state,strict=True)
    else: loader(filename=str(path),**(kwargs or {}))
    if any(not torch.equal(value.detach().cpu(),state[name]) for name,value in model.state_dict().items()):
        raise RuntimeError('native loader failed exact full checkpoint preservation')
    # Keep native grad flags for upstream DDP; enforce no_grad during each frame.
    return dict(required_states=535,state_hashes=state_hashes(model))


def main():
    parser=argparse.ArgumentParser()
    for name in ('records','report'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-channel',choices=['identity','awgn'],required=True)
    parser.add_argument('--checkpoint-sha256',required=True)
    args,remainder=parser.parse_known_args()
    if not remainder or remainder[0]!='test' or args.records.exists() or args.report.exists():
        parser.error('test-only arguments and unique evidence outputs required')
    args.records=args.records.resolve();args.report=args.report.resolve()
    args.records.parent.mkdir(parents=True,exist_ok=True);args.report.parent.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=[str(ROOT/'third_party/LIGA-Stereo'),str(ROOT/'third_party/LIGA-Stereo/tools'),str(ROOT/'third_party/mmdetection_kitti')]
    import liga.models
    original=liga.models.build_network;models=[];observers=[];loads=[];frames=[]
    report=dict(state='starting',scope='F5 fixed checkpoint evaluation and passive feature distortion',
        channel=args.expected_channel,started_at_unix=time.time(),project_sources=source_identity(ROOT,['src','scripts','configs','pyproject.toml']))
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    with args.records.open('x') as stream:
        def record(row):
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();frames.append(row['frame_id'])
        def builder(*pargs,**kwargs):
            if models: raise RuntimeError('exactly one native F5 model required')
            model=original(*pargs,**kwargs);models.append(model);native_loader=model.load_params_from_file
            def checked_loader(filename,**load_kwargs):
                if loads: raise RuntimeError('exactly one final checkpoint load required')
                loads.append(frozen_load(model,Path(filename),args.checkpoint_sha256,native_loader,load_kwargs))
            model.load_params_from_file=checked_loader
            observers.append(StereoFeatureObserver(model,args.expected_channel,record,compare_features));return model
        liga.models.build_network=builder
        try:
            sys.argv=[str(ROOT/'scripts/run_liga.py')]+remainder
            runpy.run_path(str(ROOT/'scripts/run_liga.py'),run_name='__main__')
            fold=json.loads((ROOT/'data/internal-tuning-fold-001.json').read_text())
            if frames!=fold['folds']['geocomm_tune_holdout']['ids']: raise RuntimeError('all ordered372 holdout frames required')
            if len(loads)!=1 or state_hashes(models[0])!=loads[0]['state_hashes']: raise RuntimeError('F5 model states changed during inference')
            if observers[0].calls!=dict(frames=372,student=744,codec=372,channel=372,build_cost=372,forbidden=0):
                raise RuntimeError('native F5 module counts differ')
            report.update(state='finished',frames=372,calls=observers[0].calls,checkpoint_sha256=args.checkpoint_sha256,
                checkpoint_loading=loads[0],final_state_hashes=state_hashes(models[0]),records_sha256=sha256(args.records))
        except BaseException as error:
            report.update(state='failed',exception=repr(error));raise
        finally:
            for observer in observers: observer.close()
            liga.models.build_network=original
            report['ended_at_unix']=time.time();args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__': main()
