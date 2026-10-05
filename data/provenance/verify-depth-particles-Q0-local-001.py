"""All transferred Q0 physical arrays, bounded received decode and W1 records."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def projection(values):
    """Independent scalar pool-adjacent-violators L2 ordered-cube projection."""
    output = []
    for value in values.reshape(-1, 3):
        blocks = []
        for item in value:
            blocks.append([float(item), 1])
            while len(blocks) > 1 and blocks[-2][0]/blocks[-2][1] > blocks[-1][0]/blocks[-1][1]:
                last = blocks.pop(); blocks[-1][0] += last[0]; blocks[-1][1] += last[1]
        output.append([min(1.,max(-1.,total/count)) for total,count in blocks for _ in range(count)])
    return np.array(output).reshape(values.shape)


def exact_W1(mass, edges, atoms):
    """Independent scalar integration via primitive D*abs(D)/(2*slope)."""
    mass = mass / mass.sum(); start = 0.; total = 0.
    for index, probability in enumerate(mass):
        stop = start + probability
        if probability > 0:
            slope = (edges[index+1]-edges[index])/probability
            for j, atom in enumerate(atoms):
                lower = max(start,j/3); upper = min(stop,(j+1)/3)
                if upper > lower:
                    first = edges[index]+(lower-start)*slope-atom
                    last = edges[index]+(upper-start)*slope-atom
                    total += (last*abs(last)-first*abs(first))/(2*slope)
        start = stop
    return total


def main():
    output = ROOT / 'data/provenance/depth-particles-Q0-complete-local-verification-001.json'; assert not output.exists()
    artifacts = {}; hosts = {}; all_arrays = []; raw_rows = 0
    for host in ('local','sheng'):
        folder = ROOT / 'data/engineering' / ('depth-particles-Q0-'+host+'-CPU-001')
        report_path, raw_path = folder/'report.json',folder/'complete_arrays.npz'
        report = json.loads(report_path.read_text())
        assert report['state']=='passed_Q0_synthetic_depth_particle_and_channel_feasibility_CPU' and report['host']==host
        assert report['source_distributions']==12 and report['repeats_each']==64 and report['conditions_count']==7
        assert report['no_fitted_parameters'] and report['CPU_threads']==2
        for path,digest in report['sources'].items(): assert sha(ROOT/path)==digest
        assert sha(raw_path)==report['arrays_sha256']
        for path in (report_path,raw_path):artifacts[str(path.relative_to(ROOT))]=sha(path)
        with np.load(raw_path,allow_pickle=False) as archive:
            assert sorted(archive.files)==report['arrays_keys']; arrays={name:archive[name] for name in archive.files}
        all_arrays.append(arrays)
        assert all(np.isfinite(a).all() and a.dtype in (np.dtype('<f8'),np.dtype('<c16')) for a in arrays.values())
        p,edges,atoms,q,tx,symbols = [arrays[key] for key in ('probabilities','depth_edges','particles','normalized_particles','transmitted_real','transmitted_complex')]
        assert p.shape==(12,72) and atoms.shape==q.shape==(12,3) and tx.shape==(12,4) and symbols.shape==(12,2)
        assert np.array_equal(edges,2+.8*np.arange(73)) and np.all(np.diff(atoms,axis=-1)>=0)
        assert np.max(np.abs(q-(2*(atoms-edges[0])/(edges[-1]-edges[0])-1)))<=1e-12
        expected=np.c_[q,np.ones(12)]*np.sqrt(2/(1+(q*q).sum(-1,keepdims=True)))
        assert np.max(np.abs(expected-tx))<=1e-12 and np.max(np.abs((tx*tx).sum(-1)-2))<=1e-12
        assert np.array_equal(symbols,np.stack((tx[:,0]+1j*tx[:,1],tx[:,2]+1j*tx[:,3]),-1))
        rng=np.random.Generator(np.random.PCG64(17)); noise=rng.normal(size=(64,4))
        fade=(rng.normal(size=(64,2))+1j*rng.normal(size=(64,2)))/np.sqrt(2)
        assert np.array_equal(noise,arrays['common_standard_noise']) and np.array_equal(fade,arrays['common_Rayleigh_fade'])
        intrinsic=np.array([exact_W1(row,edges,points) for row,points in zip(p,atoms)])
        assert np.max(np.abs(intrinsic-list(report['intrinsic_W1_m_by_source'].values())))<=1e-10
        errors=[]; expected_conditions=[('identity',None)]+[(channel,snr) for channel in ('awgn','rayleigh') for snr in (6,10,18)]
        assert [(item['channel'],item['SNR_dB']) for item in report['conditions']]==expected_conditions
        for item in report['conditions']:
            channel,snr=item['channel'],item['SNR_dB'];tag=channel if snr is None else channel+str(snr)
            n=np.zeros_like(noise) if snr is None else noise*np.sqrt(10**(-snr/10)/2)
            h=fade if channel=='rayleigh' else np.ones_like(fade)
            assert np.array_equal(n,arrays[tag+'_noise_real']) and np.array_equal(h,arrays[tag+'_fade'])
            complex_noise=np.stack((n[:,0]+1j*n[:,1],n[:,2]+1j*n[:,3]),-1)
            raw=symbols[:,None,:]*h[None,:,:]+complex_noise[None,:,:]
            assert np.max(np.abs(raw-arrays[tag+'_raw_received']))<=1e-12
            equalized=raw/h[None,:,:]
            received=np.stack((equalized[...,0].real,equalized[...,0].imag,equalized[...,1].real,equalized[...,1].imag),-1)
            assert np.max(np.abs(received-arrays[tag+'_equalized_real']))<=1e-12
            decoded=projection(received[...,:3]/np.maximum(received[...,3:],2**-.5))
            decoded_atoms=edges[0]+(decoded+1)*((edges[-1]-edges[0])/2)
            assert np.max(np.abs(decoded_atoms-arrays[tag+'_decoded_particles']))<=1e-12
            expected_channel=np.abs(decoded_atoms-atoms[:,None,:]).mean(-1)
            expected_full=np.array([[exact_W1(source,edges,points) for points in samples] for source,samples in zip(p,decoded_atoms)])
            errors.append(float(np.max(np.abs(expected_full-arrays[tag+'_full_histogram_W1']))))
            assert errors[-1]<=1e-10 and np.max(np.abs(expected_channel-arrays[tag+'_channel_W1']))<=1e-12
            assert item['complex_uses']==1536 and abs(item['total_energy']-1536)<=1e-10 and abs(item['mean_Es']-1)<=1e-12
            assert item['perfect_CSI']==(channel=='rayleigh') and not item['fade_clipping']
            assert len(item['rows'])==12
            for index,row in enumerate(item['rows']):
                assert abs(row['intrinsic_W1_m']-intrinsic[index])<=1e-10
                assert abs(row['channel_only_W1_mean_m']-expected_channel[index].mean())<=1e-10
                assert abs(row['full_W1_mean_m']-expected_full[index].mean())<=1e-10
                assert abs(row['full_W1_max_m']-expected_full[index].max())<=1e-10
                raw_rows+=1
        hosts[host]=dict(conditions=7,condition_source_records=84,raw_cell_records=5376,
                         maximum_independent_scalar_W1_error=max(errors))
    assert raw_rows==168 and set(all_arrays[0])==set(all_arrays[1])
    cross_host_max=max(float(np.max(np.abs(all_arrays[0][name]-all_arrays[1][name]))) for name in all_arrays[0])
    assert cross_host_max<=1e-10
    result=dict(state='passed_all10752_Q0_transferred_cell_records_and168_case_statistics',checked_unix=time.time(),
        artifacts_sha256=artifacts,hosts=hosts,cross_host_max_abs_difference=cross_host_max,
        verifier_sha256=sha(__file__),limitation='All synthetic physical arrays and received-only scalar-PAV/exact-W1 replay; no KITTI teacher, task sufficiency, GPU, learned comparator or detection advantage')
    with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(result))


if __name__=='__main__':main()
