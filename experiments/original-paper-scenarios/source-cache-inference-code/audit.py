"""Fresh native cache/prediction/calibration/AP audit without detector rerun."""
import argparse
import math
from pathlib import Path
import sys
import time

import cache_contract as c


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();assert not args.output.exists()
    run=c.read(args.manifest);scope=run['scope'];cache,ids,closure,local=c.proof(scope)
    assert run['state']=='finished' and run['ordered_ids']==ids and run['seed']==17
    assert run['sources']==c.sources() and run['protocol_sha256']==c.PROTOCOL_SHA
    assert run['inference_labels_clean_images_blocked'] and run['all_predictions_sealed_before_GT']
    assert run['source_only_PHY_uses'] is None and run['source_only_PHY_energy'] is None
    states=670 if run['detector']=='stereo_rcnn' else 484
    checkpoint=('b7a07a897224f75bc30b2d4c06f39927e92933a8bcaad6e679aff605b2346c14' if states==670 else
                '3f36af30b686f23d0868482675177b29cc50c2b378577dd12f9092ddb0235b2e')
    assert run['detector_checkpoint_sha256']==checkpoint and run['detector_states']==states
    assert len(run['detector_initial_state_hashes'])==states and run['detector_initial_state_hashes']==run['detector_final_state_hashes']
    receiver=c.read(run['cache_receiver_path']);assert c.sha(run['cache_receiver_path'])==run['cache_receiver_sha256']
    assert [r['frame_id'] for r in run['frames']]==[r['frame_id'] for r in receiver['records']]==ids
    directory=Path(run['output_dir']);predictions=directory/'data'
    assert {p.stem for p in predictions.glob('*.txt')}==set(ids)
    sys.path.insert(0,str(c.LEGACY));from common import tensor_identity
    files={};total_predictions=0
    for row,cached in zip(run['frames'],receiver['records']):
        frame=row['frame_id'];path=predictions/(frame+'.txt')
        assert row['cache_path']==cached['cache_path'] and row['cache_sha256']==cached['cache_sha256']
        assert row['received_arrays']==cached['arrays']
        # Fresh native arrays/float conversion, no native detector or alternate forward.
        outputs=c.load_pair(cached);assert [tensor_identity(x) for x in outputs]==row['received_tensors'];del outputs
        assert c.sha(path)==row['prediction_sha256']
        text=path.read_text();lines=text.splitlines();assert len(lines)==row['prediction_count']
        for line in lines:
            fields=line.split();assert len(fields)==16 and fields[0] in (('Car',) if states==670 else ('Car','Pedestrian','Cyclist'))
            values=[float(v) for v in fields[1:]];assert all(math.isfinite(v) for v in values) and 0<=values[-1]<=1
        total_predictions+=len(lines)
        if states==670:
            counts=row['native_3D_counts'];assert counts['dense_solutions']==len(lines) and row['GT_placeholder_only']
            assert row['calls']==dict(detector=1,image_backbone=2,dense_alignment=int(counts['initial_solutions']>0))
        else: assert row['calls']==dict(image_backbone=2,feature_neck=2,build_cost=1,head3D=1,forbidden=0)
        calib=c.DATA/'training/calib'/(frame+'.txt');assert c.sha(calib)==row['calibration_sha256']
        files[frame]=dict(prediction_sha256=c.sha(path),calibration_sha256=c.sha(calib),cache_sha256=c.sha(cached['cache_path']))
        if scope=='main':
            label=c.DATA/'training/label_2'/(frame+'.txt');assert c.sha(label)==row['label_sha256']
            files[frame]['label_sha256']=c.sha(label)
    if scope=='main':
        assert c.sha(directory/'metrics.json')==run['metrics_sha256'] and c.read(directory/'metrics.json')==run['metrics']
        import importlib.util
        spec=importlib.util.spec_from_file_location('source_inference_AP',c.HERE/'evaluate.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        recomputed,_=module.metrics(c.DATA,ids,predictions);assert recomputed==run['metrics']
    else: assert run['metrics'] is None;recomputed=None
    assert c.sources()==run['sources']
    result=dict(state='passed',checked_unix=time.time(),scope=scope,condition=run['condition'],detector=run['detector'],
                frames=len(ids),predictions=total_predictions,manifest_sha256=c.sha(args.manifest),
                readonly_states=states,all_received_float_tensors_recomputed=True,files=files,recomputed_metrics=recomputed,
                sources=run['sources'],scope_note='Native received-cache and official AP replay, no detector rerun; right2D absent')
    with args.output.open('x') as stream: __import__('json').dump(result,stream,indent=2,allow_nan=False)
    print(__import__('json').dumps(dict(state='passed',frames=len(ids),detector=run['detector'],condition=run['condition'])),flush=True)


if __name__=='__main__': main()
