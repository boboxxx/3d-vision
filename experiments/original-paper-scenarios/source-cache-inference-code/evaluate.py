"""Full ordered received-only native detector endpoint; predictions precede AP."""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

import cache_contract as c
sys.path.insert(0,str(c.LEGACY))
from common import tensor_identity
from receivers import LigaReceiver,StereoReceiver
from geocomm.pooling_diagnostic import state_hashes


def metrics(root,ids,directory):
    spec=importlib.util.spec_from_file_location('sealed_original_metrics',c.LEGACY/'evaluate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.metrics(root,ids,directory)


def main():
    p=argparse.ArgumentParser();p.add_argument('--scope',choices=('engineering','main'),required=True)
    p.add_argument('--condition',choices=c.CONDITIONS,required=True)
    p.add_argument('--detector',choices=('stereo_rcnn','liga'),required=True);args=p.parse_args()
    cache,ids,closure,local=c.proof(args.scope);sources=c.sources()
    name='source-cache-inference-'+args.scope+'-001-'+args.condition+'-'+args.detector
    directory=c.NATIVE/name; manifest=c.ROOT/'data/runs'/(name+'.json')
    assert not directory.exists() and not manifest.exists()
    receiver_path=cache/'received'/args.condition/'receiver.json'
    assert c.sha(receiver_path)==closure['native_files'][str(receiver_path)]
    received=c.read(receiver_path);assert received['frame_ids']==ids and received['state']=='finished_all_received_pairs'
    rows=received['records'];assert [r['frame_id'] for r in rows]==ids
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*2**20
    assert free>=12*2**30 and os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    torch.set_num_threads(2);torch.manual_seed(17);torch.cuda.manual_seed_all(17);np.random.seed(17)
    directory.mkdir();predictions=directory/'data';predictions.mkdir()
    run=dict(state='running',pid=os.getpid(),scope=args.scope,condition=args.condition,detector=args.detector,
             name=name,ordered_ids=ids,seed=17,sources=sources,frames=[],started_unix=time.time(),output_dir=str(directory),
             cache_receiver_path=str(receiver_path),cache_receiver_sha256=c.sha(receiver_path),
             cache_closure_sha256=c.sha(c.ROOT/'data/provenance'/('original-source-cache-'+args.scope+'-001-closure.json')),
             protocol_sha256=c.PROTOCOL_SHA,NVIDIA_free_before_bytes=free,source_only_PHY_uses=None,source_only_PHY_energy=None)
    c.save(manifest,run);receiver=None
    try:
        receiver=(LigaReceiver if args.detector=='liga' else StereoReceiver)()
        assert all(not m.training for m in receiver.model.modules())
        run.update(detector_checkpoint_sha256=receiver.checkpoint_sha,detector_initial_state_hashes=receiver.initial,
                   detector_states=receiver.states,construction_metadata_reads_possible=True)
        control=dict(inference_active=True);sys.addaudithook(c.guard([r['cache_path'] for r in rows],control))
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for index,row in enumerate(rows,1):
                outputs=c.load_pair(row);frame=row['frame_id']; identities=[tensor_identity(x) for x in outputs]
                calib=c.DATA/'training/calib'/(frame+'.txt')
                endpoint=receiver.predict(outputs,frame,calib,predictions);torch.cuda.synchronize()
                text=(predictions/(frame+'.txt')).read_text();lines=text.splitlines()
                assert len(lines)==endpoint['prediction_count']
                allowed=('Car',) if args.detector=='stereo_rcnn' else ('Car','Pedestrian','Cyclist')
                assert all(len(line.split())==16 and line.split()[0] in allowed and
                           np.isfinite([float(x) for x in line.split()[1:]]).all() for line in lines)
                assert state_hashes(receiver.model)==receiver.initial
                assert torch.cuda.max_memory_reserved()+2*2**30<=free
                run['frames'].append(dict(frame_id=frame,cache_path=row['cache_path'],cache_sha256=row['cache_sha256'],
                                          received_arrays=row['arrays'],received_tensors=identities,**endpoint,
                                          prediction_sha256=c.sha(predictions/(frame+'.txt')),calibration_sha256=c.sha(calib)))
                c.save(manifest,run)
                if index%20==0: print(__import__('json').dumps(dict(condition=args.condition,detector=args.detector,frames=index)),flush=True)
                del outputs
        assert len(run['frames'])==len(ids) and {p.stem for p in predictions.glob('*.txt')}==set(ids)
        # All predictions are sealed before permitting any inference label read.
        run.update(inference_labels_clean_images_blocked=True,all_predictions_sealed_before_GT=True,
                   predictions_sealed_unix=time.time());c.save(manifest,run);control['inference_active']=False
        if args.scope=='main':
            ap,text=metrics(c.DATA,ids,predictions);c.save(directory/'metrics.json',ap)
            (directory/'evaluator.txt').write_text(text)
            run.update(metrics=ap,metrics_sha256=c.sha(directory/'metrics.json'))
            for row in run['frames']: row['label_sha256']=c.sha(c.DATA/'training/label_2'/(row['frame_id']+'.txt'))
        else: run['metrics']=None
        assert c.sources()==sources and c.sha(receiver_path)==run['cache_receiver_sha256']
        run.update(state='finished',detector_final_state_hashes=state_hashes(receiver.model),ended_unix=time.time(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(),right_2D_measured=False,
                   limitation='Fixed main source-only native endpoints; no radio/matched-exposure gain or complete original matrix')
        c.save(manifest,run)
    except BaseException as error:
        run.update(state='failed',error=repr(error),ended_unix=time.time());c.save(manifest,run);raise
    finally:
        if receiver is not None: receiver.close()
    print(__import__('json').dumps(dict(state=run['state'],name=name,frames=len(ids))),flush=True)


if __name__=='__main__': main()
