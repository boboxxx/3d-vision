"""All RTX PRO6000 transferred fixtures and independent scalar native-op replay."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output=ROOT/'data/provenance/artemis-detector-operators-GPU-local-verification-001.json';assert not output.exists()
    report_path=ROOT/'data/engineering/artemis-detector-operators-GPU-001.json'
    terminal_path=ROOT/'data/engineering/artemis-detector-operators-GPU-terminal-001.json'
    report=json.loads(report_path.read_text());terminal=json.loads(terminal_path.read_text())
    assert report['state']=='passed_native_operator_GPU_only' and terminal['actual_terminal']
    assert terminal['state']=='closed_actual_Slurm_terminal_native_operator_GPU_probe'
    assert report['job_id']==terminal['job_id']=='11423980' and report['capability']==terminal['capability']==[12,0]
    assert report['GPU']==terminal['GPU']=='NVIDIA RTX PRO 6000 Blackwell Server Edition'
    assert terminal['GPU_allocation_released'] and all(row['State']=='COMPLETED' and row['ExitCode']=='0:0' for row in terminal['jobs'])
    assert terminal['closure_script_sha256']==sha(ROOT/'data/provenance/close-artemis-detector-operators-GPU-001.py')
    assert len(terminal['artifacts_sha256'])==6
    for path,digest in terminal['artifacts_sha256'].items():assert sha(ROOT/path)==digest
    assert sha(ROOT/'experiments/artemis-detector-operators/inputs-001.json')==report['input_sha256']
    build=json.loads((ROOT/'data/engineering/artemis-detector-operators-build-001.json').read_text())
    build_terminal=json.loads((ROOT/'data/engineering/artemis-detector-operators-build-terminal-001.json').read_text())
    assert sha(ROOT/'data/engineering/artemis-detector-operators-build-001.json')==report['build_manifest_sha256']
    assert build_terminal['job_id']=='11423973' and report['binaries']==build['binaries']
    assert report['packages_before']==report['packages_after']==build['packages_after']
    assert report['sources_before']==report['sources_after']
    for path,digest in {**report['execution_sources'],**report['sources_before']}.items():
        assert sha(ROOT/path)==digest==terminal['fresh_file_sha256'][path]
    for path,item in report['binaries'].items():
        assert terminal['fresh_file_sha256'][path]==item['sha256']==build_terminal['fresh_file_sha256'][path]
    folder=report_path.with_suffix('');errors={};values=0
    for ds,item in zip((1,2),report['cost_checks']):
        path=folder/('cost-downsample-'+str(ds)+'.npz')
        assert sha(path)==item['fixture_sha256']==report['artifacts'][path.name]['sha256'] and path.stat().st_size==report['artifacts'][path.name]['bytes']
        with np.load(path,allow_pickle=False) as archive:a={name:archive[name] for name in archive.files}
        left,right,shifts,g=[a[name] for name in ('left','right','shifts','incoming_gradient')]
        assert left.shape==right.shape==(1,2,6,8) and np.array_equal(shifts,np.array([[0,1,1.5]],dtype=np.float32))
        assert all(value.dtype==np.float32 and np.isfinite(value).all() for value in a.values())
        expected=np.zeros((1,4,3,6//ds,8//ds),dtype=np.float32);gl=np.zeros_like(left);gr=np.zeros_like(right)
        assert g.shape==expected.shape
        for depth,shift in enumerate(shifts[0]):
            for i in range(6//ds):
                for j in range(8//ds):
                    y,x=i*ds,j*ds;expected[0,:2,depth,i,j]=left[0,:,y,x]
                    gl[0,:,y,x]+=g[0,:2,depth,i,j]
                    pos=x-float(shift)
                    if 0<=pos<=7:
                        lo=int(np.floor(pos));hi=min(lo+1,7);weight=np.float32(pos-lo)
                        expected[0,2:,depth,i,j]=right[0,:,y,lo]*(1-weight)+right[0,:,y,hi]*weight
                        gr[0,:,y,lo]+=g[0,2:,depth,i,j]*(1-weight)
                        gr[0,:,y,hi]+=g[0,2:,depth,i,j]*weight
        current={}
        for key,literal in (('forward',expected),('left_gradient',gl),('right_gradient',gr)):
            actual,target=a['actual_'+key],a['expected_'+key]
            assert actual.shape==target.shape==literal.shape
            claimed=float(np.max(np.abs(actual-target)));assert claimed==item['max_absolute_errors'][key]
            error=float(np.max(np.abs(actual-literal)));assert error<=2e-5 and np.max(np.abs(target-literal))<=2e-5
            current[key]=error;values+=actual.size
        errors[str(ds)]=current
    path=folder/'geometric-fixtures.npz';assert sha(path)==report['geometry_checks']['fixture_sha256']
    with np.load(path,allow_pickle=False) as archive:a={name:archive[name] for name in archive.files}
    boxes=a['boxes'];assert boxes.shape==(4,7) and np.array_equal(boxes[:,6],np.zeros(4))
    centers=boxes[:,:2];size=boxes[:,3:5];lower=centers-size/2;upper=centers+size/2
    overlap=np.maximum(np.minimum(upper[:,None,:],upper[None,:,:])-np.maximum(lower[:,None,:],lower[None,:,:]),0).prod(-1)
    area=size.prod(-1);iou=overlap/(area[:,None]+area[None,:]-overlap)
    for key in ('expected_overlap','actual_overlap'):assert np.max(np.abs(a[key]-overlap))<=2e-5
    for key in ('expected_iou','actual_iou','cpu_iou'):assert np.max(np.abs(a[key]-iou))<=2e-5
    keep=[]
    for index in range(4):
        if not any(iou[index,chosen]>.5 for chosen in keep):keep.append(index)
    assert keep==a['nms_keep'].tolist()==report['geometry_checks']['NMS_keep']==[0,2,3]
    assert np.array_equal(a['roi_boxes'],boxes[[0,3]])
    membership=(np.abs(a['points'][None,:,:]-a['roi_boxes'][:,None,:3])<a['roi_boxes'][:,None,3:6]/2).all(-1).astype(np.int32)
    ids=np.where(membership.any(0),membership.argmax(0),-1)[None,:].astype(np.int32)
    assert np.array_equal(membership,a['expected_membership']) and np.array_equal(membership,a['actual_cpu_membership'])
    assert np.array_equal(ids,a['expected_gpu_ids']) and np.array_equal(ids,a['actual_gpu_ids'])
    result=dict(state='passed_complete_transferred_RTX_PRO6000_native_operator_fixtures_and_terminal',
        checked_unix=time.time(),job_id='11423980',GPU=report['GPU'],capability=[12,0],actual_terminal=True,
        artifacts_verified=7,terminal_sha256=sha(terminal_path),cost_output_gradient_values=values,
        independent_numpy_cost_max_errors=errors,BEV_overlap_IoU_NMS_ROI_membership_passed=True,
        verifier_sha256=sha(__file__),
        limitation='All probe arrays/control files and scalar CPU replay; native binary/runtime checks on Artemis, no local GPU rerun, full detector/MMCV/spconv, rotated or surface-boundary checks, training or AP')
    with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(result))


if __name__=='__main__':main()
