"""Complete transferred 24-frame received-identity/prediction/source verification."""
import hashlib
import json
import math
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]
PREFIX='source-cache-inference-engineering-001'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def mapped(p):
    path=Path(p)
    if path.is_relative_to('/home/sheng/paper6'):return ROOT/path.relative_to('/home/sheng/paper6')
    assert path.is_relative_to('/mnt/d/paper6/runs')
    return ROOT/'data/engineering'/(PREFIX+'-native-transfer')/str(path).lstrip('/')


def main():
    output=ROOT/'data/provenance'/(PREFIX+'-local-verification.json');assert not output.exists()
    closure_path=ROOT/'data/provenance'/(PREFIX+'-closure.json');closure=read(closure_path)
    assert closure['state']=='closed_all12_endpoints_actual_terminal_and_shared_received_inputs' and closure['actual_terminal']
    assert closure['scope']=='engineering' and closure['frames']==24 and closure['shared_pairs']==12
    for p,h in closure['artifacts_sha256'].items():assert sha(ROOT/p)==h,p
    assert {p:str(mapped(n).relative_to(ROOT)) for p,n in closure['native_to_local_artifacts'].items()}=={p:p for p in closure['artifacts_sha256']}
    roots=dict(project=ROOT,liga=ROOT/'third_party/LIGA-Stereo',mmdet=ROOT/'third_party/mmdetection_kitti',
               stereo_rcnn=ROOT/'third_party/Stereo-RCNN',final_evaluation=ROOT)
    source=closure['sources']
    for group,value in source['native'].items():
        assert group in roots
        hashes={p:sha(roots[group]/p) for p in value['file_hashes']};assert hashes==value['file_hashes']
        canonical=json.dumps(hashes,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        assert hashlib.sha256(canonical).hexdigest()==value['sha256']
    for p,h in source['inference'].items():assert sha(ROOT/p)==h
    assert sha(ROOT/'experiments/original-paper-scenarios/source-cache-inference-protocol-001.md')==source['protocol_sha256']
    cycle=read(ROOT/'data/runs'/(PREFIX+'-cycle.json'));launch=read(ROOT/'data/runs'/(PREFIX+'-launch.json'))
    assert cycle['pid']==launch['pid']==closure['pid']
    assert cycle['state']=='finished12_native_endpoints12_audits_pending_terminal_closure'
    assert len(cycle['completed_commands'])==24 and all(r['returncode']==0 for r in cycle['completed_commands'])
    assert sha(ROOT/'data/provenance/source-cache-inference-cycle-001.py')==launch['controller_sha256']==cycle['controller_sha256']
    conditions=[f'{codec}-cr{rate}' for codec in ('jpeg','jpeg2000') for rate in (10,30,50)]
    expected={condition+'-'+detector for condition in conditions for detector in ('stereo_rcnn','liga')}
    assert set(cycle['endpoints'])==set(closure['endpoints'])==expected
    frames=0;shared={};predictions=0
    for label,item in cycle['endpoints'].items():
        manifest=mapped(item['manifest']);audit=mapped(item['audit']);run=read(manifest);report=read(audit)
        assert sha(manifest)==item['manifest_sha256']==report['manifest_sha256'] and sha(audit)==item['audit_sha256']
        assert run['state']=='finished' and report['state']=='passed' and report['frames']==2
        assert run['sources']==report['sources']==source and run['scope']==report['scope']=='engineering'
        assert run['metrics'] is report['recomputed_metrics'] is None
        assert run['ordered_ids']==['000000','000003'] and [r['frame_id'] for r in run['frames']]==run['ordered_ids']
        assert run['all_predictions_sealed_before_GT'] and run['inference_labels_clean_images_blocked']
        states=670 if run['detector']=='stereo_rcnn' else 484
        assert len(run['detector_initial_state_hashes'])==states and run['detector_initial_state_hashes']==run['detector_final_state_hashes']
        assert report['readonly_states']==run['detector_states']==states
        for row in run['frames']:
            frame=row['frame_id'];path=mapped(str(Path(run['output_dir'])/'data'/(frame+'.txt')))
            assert sha(path)==row['prediction_sha256']==report['files'][frame]['prediction_sha256']
            lines=path.read_text().splitlines();assert len(lines)==row['prediction_count']
            for line in lines:
                fields=line.split();assert len(fields)==16 and fields[0] in (('Car',) if states==670 else ('Car','Pedestrian','Cyclist'))
                values=[float(x) for x in fields[1:]];assert all(math.isfinite(x) for x in values) and 0<=values[-1]<=1
            assert row['calibration_sha256']==report['files'][frame]['calibration_sha256']
            assert row['cache_sha256']==report['files'][frame]['cache_sha256']
            if states==670:
                assert row['calls']==dict(detector=1,image_backbone=2,dense_alignment=int(row['native_3D_counts']['initial_solutions']>0))
                assert row['native_3D_counts']['dense_solutions']==len(lines) and row['GT_placeholder_only']
            else:assert row['calls']==dict(image_backbone=2,feature_neck=2,build_cost=1,head3D=1,forbidden=0)
            value=dict(cache_sha256=row['cache_sha256'],arrays=row['received_arrays'],tensors=row['received_tensors'])
            key=(run['condition'],frame);assert key not in shared or shared[key]==value;shared[key]=value
            for tensor in row['received_tensors']:
                assert tensor['dtype']=='float32' and tensor['shape'][:2]==[1,3]
                assert 0<=tensor['minimum']<=tensor['maximum']<=1 and tensor['out_of_range_fraction']==0
            frames+=1;predictions+=len(lines)
    assert frames==24 and len(shared)==12
    result=dict(state='passed_all24_transferred_native_inference_records_and_predictions',checked_unix=time.time(),frames=frames,
                predictions=predictions,sources=source,artifacts_verified=len(closure['artifacts_sha256']),
                closure_sha256=sha(closure_path),shared_pairs=12,AP_measured=False,
                limitation='All received arrays/float conversion and states checked natively; local complete records/prediction checks, no local pixel or GPU replay')
    with output.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(dict(state=result['state'],frames=frames,predictions=predictions)))


if __name__=='__main__':main()
