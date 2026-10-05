"""Independent local NumPy audit of raw GPU fixtures and sealed Slurm evidence."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ENGINEERING = ROOT / 'data/engineering'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cost_reference(f, ds):
    left, right, shifts, incoming = (f[k] for k in ('left', 'right', 'shifts', 'incoming_gradient'))
    assert left.shape == right.shape == (1,2,6,8)
    assert np.array_equal(shifts, [[0,1,1.5]])
    expected = np.zeros_like(f['actual_forward'])
    gl, gr = np.zeros_like(left), np.zeros_like(right)
    _, channels, h, w = left.shape
    for c in range(channels):
        for d, shift in enumerate(shifts[0]):
            for y in range(h // ds):
                for x in range(w // ds):
                    yy, xx = y * ds, x * ds
                    expected[0,c,d,y,x] = left[0,c,yy,xx]
                    gl[0,c,yy,xx] += incoming[0,c,d,y,x]
                    pos = xx - float(shift)
                    if pos < 0 or pos > w-1:
                        continue
                    lo = int(np.floor(pos))
                    hi = min(lo+1,w-1)
                    weight = pos-lo
                    expected[0,c+channels,d,y,x] = right[0,c,yy,lo]*(1-weight) + right[0,c,yy,hi]*weight
                    g = incoming[0,c+channels,d,y,x]
                    gr[0,c,yy,lo] += g*(1-weight)
                    gr[0,c,yy,hi] += g*weight
    errors = {}
    for name, target in [('forward',expected), ('left_gradient',gl), ('right_gradient',gr)]:
        for prefix in ('actual_', 'expected_'):
            observed = f[prefix+name]
            assert np.isfinite(observed).all()
            np.testing.assert_allclose(observed,target,atol=2e-5,rtol=2e-5)
            errors[prefix+name] = float(np.max(np.abs(observed-target)))
    return errors


def main():
    lock = json.loads((ROOT/'experiments/artemis-detector-operators/inputs-001.json').read_text())
    build_path = ENGINEERING/'artemis-detector-operators-build-001.json'
    probe_path = ENGINEERING/'artemis-detector-operators-GPU-001.json'
    terminal = json.loads((ENGINEERING/'artemis-detector-operators-terminal-001.json').read_text())
    build, probe = json.loads(build_path.read_text()), json.loads(probe_path.read_text())
    assert build['state'] == 'passed_build_only_GPU_unverified'
    assert probe['state'] == 'passed_native_operator_GPU_only'
    assert probe['GPU'] == 'NVIDIA RTX PRO 6000 Blackwell Server Edition' and probe['capability'] == [12,0]
    assert probe['torch_version'] == '2.7.1+cu128' and probe['torch_cuda'] == '12.8'
    assert probe['build_manifest_sha256'] == sha(build_path)
    assert build['packages_before'] == build['packages_after'] == probe['packages_before'] == probe['packages_after']
    local_sources = {name: sha(ROOT/name) for name in lock['source_files']}
    assert local_sources == lock['source_files'] == build['sources_before'] == build['sources_after'] == probe['sources_before'] == probe['sources_after']
    for name, comp in lock['components'].items():
        assert build['components'][name] == {'archive_sha256':comp['sha256'], 'bytes':int(comp['size'])}
    assert build['binaries'] == probe['binaries']
    for job, needs_gpu in [(build['job_id'],False),(probe['job_id'],True)]:
        rows = terminal['jobs'][job]
        assert rows and all(r['State']=='COMPLETED' and r['ExitCode']=='0:0' for r in rows)
        allocation = next(r for r in rows if r['JobID']==job)
        assert ('gres/gpu' in allocation['AllocTRES']) == needs_gpu
    for name, expected in terminal['fresh_file_sha256'].items():
        if name.startswith('data/engineering/'):
            assert sha(ROOT/name) == expected
    for name, rec in probe['binaries'].items():
        assert terminal['fresh_file_sha256'][name] == rec['sha256']
    directory = probe_path.with_suffix('')
    for name, rec in probe['artifacts'].items():
        p=directory/name
        assert p.stat().st_size==rec['bytes'] and sha(p)==rec['sha256']
    checks={}
    for ds in (1,2):
        with np.load(directory/f'cost-downsample-{ds}.npz',allow_pickle=False) as f:
            checks[f'cost_downsample_{ds}'] = cost_reference(f,ds)
    with np.load(directory/'geometric-fixtures.npz',allow_pickle=False) as f:
        b=f['boxes']; lower=b[:,:2]-b[:,3:5]/2;upper=b[:,:2]+b[:,3:5]/2
        intersection=np.maximum(np.minimum(upper[:,None],upper[None])-np.maximum(lower[:,None],lower[None]),0).prod(axis=-1)
        area=b[:,3]*b[:,4];iou=intersection/(area[:,None]+area[None]-intersection)
        for field in ('expected_overlap','actual_overlap'):np.testing.assert_allclose(f[field],intersection,atol=2e-5,rtol=2e-5)
        for field in ('expected_iou','actual_iou','cpu_iou'):np.testing.assert_allclose(f[field],iou,atol=2e-5,rtol=2e-5)
        remaining=list(range(4));keep=[]
        while remaining:
            first=remaining.pop(0);keep.append(first);remaining=[k for k in remaining if iou[first,k]<=.5]
        np.testing.assert_array_equal(f['nms_keep'],keep)
        membership=(np.abs(f['points'][None]-f['roi_boxes'][:,None,:3])<f['roi_boxes'][:,None,3:6]/2).all(axis=-1).astype(np.int32)
        ids=np.full((1,len(f['points'])),-1,np.int32)
        for i,row in enumerate(membership):ids[0,row.astype(bool)]=i
        for field in ('expected_membership','actual_cpu_membership'):np.testing.assert_array_equal(f[field],membership)
        for field in ('expected_gpu_ids','actual_gpu_ids'):np.testing.assert_array_equal(f[field],ids)
        checks['IoU_NMS_ROI']='passed_independent_coordinate_bounds'
    output=ROOT/'data/provenance/artemis-detector-operators-001-local-verification.json'
    assert not output.exists()
    output.write_text(json.dumps({'state':'passed','evidence_scope':'native_operator_engineering_no_detector_loss_or_AP',
                                 'verifier_sha256':sha(Path(__file__)), 'build_manifest_sha256':sha(build_path),
                                 'probe_manifest_sha256':sha(probe_path), 'terminal_manifest_sha256':sha(ENGINEERING/'artemis-detector-operators-terminal-001.json'),
                                 'checks':checks},indent=2)+'\n')
    print('passed independent raw fixture, source, binary and actual terminal verification')


if __name__=='__main__':main()
