"""Q0 fixed synthetic geometry/physical accounting and intrinsic W1 experiment."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize
import torch

import representation as r

ROOT = Path(__file__).resolve().parents[3]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numpy_particles(p, edges):
    result = []
    for row in p:
        mass = row / row.sum(); cumulative = np.cumsum(mass); values = []
        for level in (1/6, 1/2, 5/6):
            index = int(np.searchsorted(cumulative, level))
            before = cumulative[index - 1] if index else 0.
            values.append(edges[index] + (level - before) / mass[index] * (edges[index + 1] - edges[index]))
        result.append(values)
    return np.array(result)


def source_distributions():
    names = ['narrow_near', 'narrow_middle', 'narrow_far', 'uniform', 'broad_middle', 'broad_far',
             'bimodal_50_50', 'bimodal_35_65', 'far_minority_5pct', 'far_minority_10pct',
             'far_minority_20pct', 'far_minority_40pct']
    edges = 2 + .8 * np.arange(73); centers = (edges[:-1] + edges[1:]) / 2
    values = []
    for index in (7, 35, 67):
        value = np.zeros(72); value[index] = 1; values.append(value)
    values.append(np.ones(72) / 72)
    for center, sigma in ((30, 5), (50, 8)):
        value = np.exp(-.5 * ((centers - center) / sigma)**2); values.append(value / value.sum())
    for far_mass in (.5, .65, .05, .10, .20, .40):
        value = np.zeros(72); value[7] = 1 - far_mass; value[67] = far_mass; values.append(value)
    assert len(values) == len(names) == 12
    return names, np.array(values), edges


def checks(p, edges):
    tensor = torch.from_numpy(p); bounds = torch.from_numpy(edges)
    atoms = r.particles(tensor, bounds)
    error = float(np.max(np.abs(atoms.numpy() - numpy_particles(p, edges)))); assert error <= 1e-12
    q = r.scale_depth(atoms, edges[0], edges[-1]); tx = r.encode(q)
    inversion = float((r.decode(tx) - q).abs().max()); assert inversion <= 1e-12
    energy_error = float((tx.square().sum(-1) - 2).abs().max()); assert energy_error <= 1e-12
    rng = np.random.Generator(np.random.PCG64(17))
    probe = rng.normal(0, 3, size=(32, 3)); projected = r.project_ordered_cube(torch.from_numpy(probe)).numpy()
    optimizer_error = 0.; constraint = LinearConstraint(np.array([[-1.,1.,0.],[0.,-1.,1.]]), 0, np.inf)
    for value, actual in zip(probe, projected):
        result = minimize(lambda x: .5 * np.sum((x-value)**2), np.zeros(3), jac=lambda x:x-value,
            bounds=Bounds(-np.ones(3), np.ones(3)), constraints=[constraint], method='SLSQP',
            options={'ftol':1e-13, 'maxiter':200})
        assert result.success, result.message
        optimizer_error = max(optimizer_error, float(np.max(np.abs(actual-result.x))))
    assert optimizer_error <= 2e-7
    grid = torch.tensor(list(itertools.combinations_with_replacement(np.linspace(-1,1,11),3)), dtype=torch.float64)
    assert len(grid) == 286
    noise = torch.from_numpy(rng.normal(0,2,size=(286,32,4)))
    received = r.encode(grid)[:,None,:] + noise
    reconstructed = r.decode(received)
    norms = (reconstructed - grid[:,None,:]).norm(dim=-1); bound = 2 * np.sqrt(2) * noise.norm(dim=-1)
    maximum_excess = float((norms-bound).max()); assert maximum_excess <= 2e-12
    positive = torch.from_numpy(rng.uniform(.5,1.5,size=(2,72))).requires_grad_()
    quantile_gradient = torch.autograd.gradcheck(lambda x:r.particles(x,bounds), (positive,), eps=1e-6, atol=2e-7, rtol=1e-5)
    coordinates = torch.tensor([[-.7,-.1,.6],[-.5,.2,.8]], dtype=torch.float64, requires_grad=True)
    carrier_gradient = torch.autograd.gradcheck(lambda x:r.decode(r.encode(x)), (coordinates,), eps=1e-6, atol=1e-7, rtol=1e-5)
    assert quantile_gradient and carrier_gradient
    # Independent analytic values for uniform/narrow intervals and midpoint
    # integration of inverse CDF for every source (not the exact integration code).
    intrinsic = r.histogram_W1(p, edges, atoms.numpy())
    assert np.max(np.abs(intrinsic[:3] - .8/12)) <= 1e-12 and abs(intrinsic[3] - 57.6/12) <= 1e-12
    levels = (np.arange(120000) + .5) / 120000; numeric_errors = []
    for row, points, exact in zip(p, atoms.numpy(), intrinsic):
        cumulative = row.cumsum(); index = np.searchsorted(cumulative, levels)
        previous = np.r_[0,cumulative[:-1]][index]
        truth = edges[index] + (levels-previous)/row[index] * (edges[index+1]-edges[index])
        approximation = points[np.minimum((levels*3).astype(int),2)]
        numeric_errors.append(abs(np.mean(np.abs(truth-approximation))-exact))
    assert max(numeric_errors) <= 2e-5
    return dict(quantile_reference_max_error=error, noiseless_max_error=inversion,
        per_cell_energy_max_error=energy_error, projection_optimizer_max_error=optimizer_error,
        projection_optimizer_cases=32, bound_cases=9152, maximum_noise_bound_excess=maximum_excess,
        quantile_gradient_passed=quantile_gradient, carrier_gradient_passed=carrier_gradient,
        W1_numeric_reference_max_error=max(numeric_errors)), atoms.numpy(), q.numpy(), tx.numpy(), intrinsic


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--host', choices=('local','sheng'), required=True); args=parser.parse_args()
    torch.set_num_threads(2)
    folder = ROOT / 'data/engineering' / ('depth-particles-Q0-' + args.host + '-CPU-001')
    assert not folder.exists(); folder.mkdir()
    started = time.time(); names,p,edges=source_distributions(); checks_report,atoms,q,tx,intrinsic=checks(p,edges)
    rng=np.random.Generator(np.random.PCG64(17)); common_noise=rng.normal(size=(64,4))
    common_fade=(rng.normal(size=(64,2))+1j*rng.normal(size=(64,2)))/np.sqrt(2)
    assert (np.abs(common_fade)>0).all()
    source_symbols=np.stack((tx[:,0]+1j*tx[:,1],tx[:,2]+1j*tx[:,3]),-1)
    arrays=dict(probabilities=p, depth_edges=edges, particles=atoms, normalized_particles=q, transmitted_real=tx,
                transmitted_complex=source_symbols, common_standard_noise=common_noise, common_Rayleigh_fade=common_fade)
    conditions=[]
    for channel,snr in [('identity',None)]+[(channel,snr) for channel in ('awgn','rayleigh') for snr in (6,10,18)]:
        tag=channel if snr is None else channel+str(snr)
        noise=np.zeros_like(common_noise) if snr is None else common_noise*np.sqrt(10**(-snr/10)/2)
        fade=np.ones_like(common_fade) if channel!='rayleigh' else common_fade
        complex_noise=np.stack((noise[:,0]+1j*noise[:,1],noise[:,2]+1j*noise[:,3]),-1)
        raw_received=source_symbols[:,None,:]*fade[None,:,:]+complex_noise[None,:,:]
        equalized=raw_received/fade[None,:,:]
        received=np.stack((equalized[...,0].real,equalized[...,0].imag,equalized[...,1].real,equalized[...,1].imag),-1)
        decoded=r.decode(torch.from_numpy(received)).numpy(); decoded_atoms=r.unscale_depth(decoded,edges[0],edges[-1])
        channel_W1=np.abs(decoded_atoms-atoms[:,None,:]).mean(-1)
        full_W1=r.histogram_W1(p[:,None,:],edges,decoded_atoms)
        effective_noise=received-tx[:,None,:]; discrepancy=np.linalg.norm(decoded-q[:,None,:],axis=-1)
        bound=2*np.sqrt(2)*np.linalg.norm(effective_noise,axis=-1)
        assert np.max(discrepancy-bound)<=2e-12
        assert np.max(full_W1-intrinsic[:,None]-channel_W1)<=2e-12
        assert np.isfinite(decoded).all() and ((decoded>=-1)&(decoded<=1)).all()
        energy=float(np.sum(np.abs(source_symbols)**2)*64); uses=12*64*2
        assert abs(energy-uses)<=1e-10
        arrays.update({tag+'_noise_real':noise,tag+'_fade':fade,tag+'_raw_received':raw_received,
                       tag+'_equalized_real':received,tag+'_decoded_particles':decoded_atoms,
                       tag+'_channel_W1':channel_W1,tag+'_full_histogram_W1':full_W1})
        rows=[dict(name=name,intrinsic_W1_m=float(base),channel_only_W1_mean_m=float(chan.mean()),
                   full_W1_mean_m=float(full.mean()),full_W1_max_m=float(full.max()))
              for name,base,chan,full in zip(names,intrinsic,channel_W1,full_W1)]
        conditions.append(dict(channel=channel,SNR_dB=snr,complex_uses=uses,total_energy=energy,mean_Es=energy/uses,
            perfect_CSI=channel=='rayleigh',fade_clipping=False,rows=rows))
    raw_path=folder/'complete_arrays.npz'; np.savez(raw_path,**arrays)
    sources={str(path.relative_to(ROOT)):sha(path) for path in [ROOT/'experiments/depth-particles/protocol-001.md',*Path(__file__).resolve().parent.glob('*.py')]}
    result=dict(state='passed_Q0_synthetic_depth_particle_and_channel_feasibility_CPU',host=args.host,
        started_unix=started,ended_unix=time.time(),sources=sources,checks=checks_report,conditions=conditions,
        source_distributions=12,repeats_each=64,conditions_count=7,arrays_sha256=sha(raw_path),arrays_keys=sorted(arrays),
        intrinsic_W1_m_by_source={name:float(value) for name,value in zip(names,intrinsic)},
        no_fitted_parameters=True,CPU_threads=2,torch_version=torch.__version__,numpy_version=np.__version__,
        limitation='Synthetic histogram/charged two-symbol geometry transport only; no native teacher calibration/sufficiency, appearance/detector, KITTI, GPU, full task rate-distortion or AP')
    with (folder/'report.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(state=result['state'],checks=checks_report,intrinsic_W1_m=result['intrinsic_W1_m_by_source'])))


if __name__=='__main__':main()
