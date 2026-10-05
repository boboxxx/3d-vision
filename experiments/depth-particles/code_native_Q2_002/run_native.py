"""All-grid CPU native-cache diagnostic; only allowlisted NPZ entries are read."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import representation as r

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'data/engineering/depth-particles-native-Q2-sheng-CPU-002'
FIXTURES = Path('/mnt/d/paper6/runs/geometry-risk-native-engineering-001')
LOCK = {
    '000000': ('cbb10d56055ec6c540b5f103d25a904e67d80c06300580fe6ee3104fa927fd17', 118431696),
    '000003': ('402f4c91fc9a26100d8e20f14a16b6b9215a65d93d39858d6986ad8299062a58', 117746968),
}
KEYS = ['left_probability', 'right_probability', 'left_valid', 'right_valid',
        'disparity_bins', 'P2', 'P3', 'valid_image_shape']

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

q0 = module('q0_fixed', 'experiments/depth-particles/code/representation.py')
q1 = module('q1_fixed', 'experiments/depth-particles/code_Q1/representation.py')

def save(report):
    (OUT/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

def stats(values, valid):
    x = values[valid]
    return dict(mean_W1_m=float(x.mean()), median_W1_m=float(np.median(x)),
                max_W1_m=float(x.max()), p95_W1_m=float(np.quantile(x, .95)), count=len(x))

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(2)
    assert not OUT.exists(); OUT.mkdir(parents=True)
    paths = ['experiments/depth-particles/protocol-003.md', 'experiments/depth-particles/protocol-004.md',
        'experiments/depth-particles/code_native_Q2_002/representation.py',
        'experiments/depth-particles/code_native_Q2_002/run_native.py',
        'experiments/depth-particles/code/representation.py',
        'experiments/depth-particles/code_Q1/representation.py',
        'experiments/geometry-risk/code/proxy.py',
        'third_party/LIGA-Stereo/liga/utils/calibration_kitti.py',
        'data/provenance/geometry-risk-native-engineering-001-audit.json']
    sources = {p: sha(ROOT/p) for p in paths}
    report = dict(state='running', started_unix=time.time(), sources=sources,
        host='sheng', read_keys=KEYS, GPU_used=False, bounds_m=[0.,128.], seed=2806,
        conditions=['identity','awgn10'], methods=['Q0','Q1','Q2'], views=[], fixtures={},
        source_type='uncalibrated_cosine_epipolar_F6b_train_cache_not_author_depth_head',
        task_or_true_depth_or_calibration_or_AP_claim=False)
    save(report); rng = np.random.Generator(np.random.PCG64(2806))
    for frame, (digest, size) in LOCK.items():
        fixture = FIXTURES/(frame+'-clean-fixture.npz')
        assert fixture.stat().st_size == size and sha(fixture) == digest
        with np.load(fixture, allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in KEYS}
        assert sha(fixture) == digest
        assert data['P2'].dtype == data['P3'].dtype == np.float32
        fb = float(np.abs(data['P2'][0,3]-data['P3'][0,3]))
        assert np.array_equal(data['disparity_bins'], 4*np.arange(1,49))
        z = (fb/data['disparity_bins'])[::-1].copy()
        assert (np.diff(z)>0).all() and 0 < z[0] < z[-1] < 128
        report['fixtures'][frame] = dict(path=str(fixture), sha256_before=digest,
            sha256_after=digest, bytes=size, focal_baseline=fb)
        for view in ('left','right'):
            tag = frame+'-'+view; start = time.time()
            p = data[view+'_probability'][::-1].reshape(48,-1).T.copy()
            valid = data[view+'_valid'].reshape(-1).copy(); n = len(valid)
            assert p.shape == (24960,48) and valid.dtype == np.bool_
            assert np.isfinite(p).all() and (p>=0).all()
            assert np.max(np.abs(p.sum(-1)[valid]-1)) < 2e-14
            assert (p[~valid] == 0).all()
            pn = p[valid]/p[valid].sum(-1,keepdims=True)
            atoms0 = np.full((n,3),64.); atoms1 = np.full((n,2),64.); mass1 = np.full(n,.5)
            best = np.full(n,-1,dtype=np.int16); candidates = np.full((n,48),np.nan)
            costs = np.full((n,48),np.nan)
            a,m,b,cm,cv = r.fit(pn,z)
            atoms0[valid] = r.equal_quantiles(pn,z); atoms1[valid] = a; mass1[valid] = m
            best[valid]=b; candidates[valid]=cm; costs[valid]=cv
            weights0 = np.full((n,3),1/3); weights1 = np.stack((mass1,1-mass1),-1)
            intrinsic0 = np.full(n,np.nan); intrinsic1 = np.full(n,np.nan)
            intrinsic0[valid] = r.discrete_W1(pn,z,atoms0[valid],weights0[valid])
            intrinsic1[valid] = r.discrete_W1(pn,z,a,weights1[valid])
            assert np.max(np.abs(intrinsic1[valid]-cv.min(-1))) < 2e-11
            coordinates = {'Q0':2*atoms0/128-1,
                'Q1':q1.coordinates(atoms1,mass1,0,128),
                'Q2':2*r.barycentric(atoms1,mass1)/128-1}
            standard = rng.normal(size=(n,4)); noise=standard*np.sqrt(.05)
            arrays = dict(probability=p,valid=valid,depth_support=z,P2=data['P2'],P3=data['P3'],
                disparity_bins=data['disparity_bins'],valid_image_shape=data['valid_image_shape'],
                candidate_mass=candidates,candidate_W1=costs,selected_partition=best,
                Q0_source_atoms=atoms0,Q1_source_atoms=atoms1,Q1_source_mass=mass1,
                Q0_intrinsic_W1=intrinsic0,Q1_intrinsic_W1=intrinsic1,standard_noise=standard)
            entry = dict(tag=tag,grid_cells=n,valid_cells=int(valid.sum()),invalid_cells=int((~valid).sum()),
                methods={},source_intrinsic={'Q0':stats(intrinsic0,valid),'Q1':stats(intrinsic1,valid)})
            for method in report['methods']:
                q = coordinates[method]; codec = q1 if method=='Q1' else q0
                tx = codec.encode(torch.from_numpy(q)).numpy()
                assert np.max(np.abs((tx*tx).sum(-1)-2)) < 2e-14
                arrays[method+'_coordinates']=q; arrays[method+'_tx_real']=tx
                source_atoms=atoms0 if method=='Q0' else atoms1
                source_weights=weights0 if method=='Q0' else weights1
                intrinsic=intrinsic0 if method=='Q0' else intrinsic1
                entry['methods'][method]={}
                for condition in report['conditions']:
                    recv=tx+(0 if condition=='identity' else noise)
                    dec=codec.decode(torch.from_numpy(recv)).numpy()
                    if method=='Q0': atoms=(dec+1)*64; weights=weights0
                    elif method=='Q1':
                        atoms,m=q1.unpack(dec,0,128); weights=np.stack((m,1-m),-1)
                    else:
                        atoms,m=r.recover_barycentric((dec+1)*64); weights=np.stack((m,1-m),-1)
                    full=np.full(n,np.nan); channel=np.full(n,np.nan)
                    full[valid]=r.discrete_W1(pn,z,atoms[valid],weights[valid])
                    channel[valid]=r.discrete_W1(source_weights[valid],source_atoms[valid],atoms[valid],weights[valid])
                    assert np.max(full[valid]-intrinsic[valid]-channel[valid]) < 3e-11
                    effective=np.linalg.norm(recv-tx,axis=-1)
                    assert np.max(np.linalg.norm(dec-q,axis=-1)-2*np.sqrt(2)*effective) < 3e-14
                    if method=='Q1': bound=256*effective
                    elif method=='Q2': bound=np.sqrt(10)*128*effective
                    else: bound=128*np.sqrt(2/3)*effective
                    assert np.max(channel[valid]-bound[valid]) < 3e-11
                    if condition=='identity': assert np.max(np.abs(dec-q)) < 2e-14 and channel[valid].max()<3e-11
                    prefix=condition+'_'+method
                    arrays.update({prefix+'_received_real':recv,prefix+'_decoded_coordinates':dec,
                        prefix+'_atoms':atoms,prefix+'_weights':weights,prefix+'_full_W1':full,
                        prefix+'_channel_W1':channel})
                    entry['methods'][method][condition]=dict(full=stats(full,valid),channel=stats(channel,valid),
                        complex_uses=n*2,total_energy=float((tx*tx).sum()),mean_Es=float((tx*tx).sum()/(n*2)))
            entry['Q2_minus_Q1_AWGN10'] = stats(arrays['awgn10_Q2_full_W1']-arrays['awgn10_Q1_full_W1'],valid)
            entry['Q2_AWGN10_better_fraction']=float(np.mean(arrays['awgn10_Q2_full_W1'][valid]<arrays['awgn10_Q1_full_W1'][valid]))
            dest=OUT/(tag+'.npz'); np.savez_compressed(dest,**arrays)
            entry.update(artifact=str(dest.relative_to(ROOT)),sha256=sha(dest),bytes=dest.stat().st_size,seconds=time.time()-start)
            report['views'].append(entry); save(report)
            print(json.dumps(dict(tag=tag,valid=entry['valid_cells'],paired=entry['Q2_minus_Q1_AWGN10']['mean_W1_m'])),flush=True)
    assert sources == {p: sha(ROOT/p) for p in paths}
    report.update(state='passed_complete_native_cached_posterior_transport_pending_independent_local_verification',
        ended_unix=time.time(),grid_cells=sum(v['grid_cells'] for v in report['views']),
        valid_cells=sum(v['valid_cells'] for v in report['views']),
        attempted_complex_uses=sum(v['grid_cells'] for v in report['views'])*2*6)
    assert report['attempted_complex_uses']==1198080
    save(report)

if __name__=='__main__': main()
