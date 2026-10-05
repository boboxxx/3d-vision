#!/usr/bin/env python3
"""Full native F9 evaluator: exact539 load, passive coding/feature/receiver evidence."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(ROOT/'src'))
from common import frozen_load,treatment,F9Observer,evaluation_sources,validate_native_record,install_native3D_forward
from geocomm.pooling_diagnostic import state_hashes
from geocomm.evidence import sha256,source_identity
from diagnose_codec_features import compare_features


def main():
    parser=argparse.ArgumentParser()
    for name in ('records','report'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-channel',choices=['identity','awgn'],required=True)
    parser.add_argument('--checkpoint-sha256',required=True)
    parser.add_argument('--arm',choices=('U','G','P','S'),required=True)
    parser.add_argument('--expected-snr',type=float,choices=(6.,10.,18.),required=True)
    args,remainder=parser.parse_known_args()
    if not remainder or remainder[0]!='test' or args.records.exists() or args.report.exists():
        parser.error('test-only arguments and unique evidence outputs required')
    args.records=args.records.resolve();args.report=args.report.resolve()
    args.records.parent.mkdir(parents=True,exist_ok=True);args.report.parent.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=[str(ROOT/'third_party/LIGA-Stereo'),str(ROOT/'third_party/LIGA-Stereo/tools'),str(ROOT/'third_party/mmdetection_kitti')]
    import liga.models
    original=liga.models.build_network;models=[];observers=[];loads=[];frames=[]
    report=dict(state='starting',scope='F9 final539-state evaluation, passive received-feature/coding/RNG diagnostics',
        arm=args.arm,channel=args.expected_channel,snr_db=args.expected_snr,started_at_unix=time.time(),evaluation_sources=evaluation_sources(),project_sources=source_identity(ROOT,['src','scripts','configs','pyproject.toml']))
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    with args.records.open('x') as stream:
        def record(row):
            validate_native_record(row,args.arm,args.expected_channel,args.expected_snr)
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();frames.append(row['frame_id'])
        def builder(*pargs,**kwargs):
            if models: raise RuntimeError('exactly one native F9 model required')
            model=original(*pargs,**kwargs);treatment(model,args.arm,args.expected_channel,args.expected_snr);install_native3D_forward(model);models.append(model);native_loader=model.load_params_from_file
            def checked_loader(filename,**load_kwargs):
                if loads: raise RuntimeError('exactly one final checkpoint load required')
                loads.append(frozen_load(model,Path(filename),args.checkpoint_sha256,native_loader,load_kwargs))
            model.load_params_from_file=checked_loader
            observers.append(F9Observer(model,args.arm,args.expected_channel,args.expected_snr,record));return model
        liga.models.build_network=builder
        try:
            sys.argv=[str(ROOT/'scripts/run_liga.py')]+remainder
            runpy.run_path(str(ROOT/'scripts/run_liga.py'),run_name='__main__')
            fold=json.loads((ROOT/'data/internal-tuning-fold-001.json').read_text())
            if frames!=fold['folds']['geocomm_tune_holdout']['ids']: raise RuntimeError('all ordered372 holdout frames required')
            if len(loads)!=1 or state_hashes(models[0])!=loads[0]['state_hashes']: raise RuntimeError('F9 model states changed during inference')
            if observers[0].calls!=dict(frames=372,student=744,codec=372,channel=372,build_cost=372,forbidden=0):
                raise RuntimeError('native F9 module counts differ')
            if evaluation_sources()!=report['evaluation_sources']:raise RuntimeError('evaluation sources changed during endpoint')
            report.update(state='finished',frames=372,calls=observers[0].calls,checkpoint_sha256=args.checkpoint_sha256,
                checkpoint_loading=loads[0],receiver_execution=models[0].F9_receiver_execution,final_state_hashes=state_hashes(models[0]),records_sha256=sha256(args.records))
        except BaseException as error:
            report.update(state='failed',exception=repr(error));raise
        finally:
            for observer in observers: observer.close()
            liga.models.build_network=original
            report['ended_at_unix']=time.time();args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__': main()
