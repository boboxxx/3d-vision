"""Independent enumeration, received-only integration and declared-prior risk."""
import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import time
import sys
import traceback
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'experiments/cost-field/ordered-receiver-code-015'))
from receiver import likelihood, fading_blind_likelihood, marginals, isotonic_three, decode

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def exact_checks():
    grid = torch.linspace(0, 1, 9, dtype=torch.float64)
    triples = np.asarray(list(itertools.combinations_with_replacement(range(9), 3)))
    rng = np.random.default_rng(1715)
    ll = np.concatenate([np.zeros((1, 3, 9)), rng.normal(size=(31, 3, 9))*30])
    actual = marginals(torch.from_numpy(ll)).numpy()
    expected = np.zeros_like(actual)
    for n, scores in enumerate(ll):
        weights = np.exp(sum(scores[j, triples[:, j]] for j in range(3))-
            max(sum(scores[j, triples[:, j]] for j in range(3))))
        weights /= weights.sum()
        for j in range(3):
            np.add.at(expected[n, j], triples[:, j], weights)
    assert np.max(np.abs(actual-expected)) <= 5e-12
    assert np.max(np.abs(actual.sum(-1)-1)) <= 5e-12
    means = actual@grid.numpy()
    assert np.min(np.diff(means, axis=-1)) >= -5e-12
    # Brute partitions independently validate the weighted isotonic control.
    x = rng.uniform(0, 1, (128, 3)); w = np.exp(rng.normal(size=(128, 3))*2)
    got = isotonic_three(torch.from_numpy(x), torch.from_numpy(w)).numpy()
    wanted = []
    for values, weights in zip(x, w):
        candidates = []
        for groups in [((0,), (1,), (2,)), ((0, 1), (2,)), ((0,), (1, 2)), ((0, 1, 2),)]:
            trial = np.zeros(3)
            for group in groups:
                ix = list(group); trial[ix] = np.average(values[ix], weights=weights[ix])
            if (np.diff(trial) >= 0).all():
                candidates.append((np.sum(weights*(values-trial)**2), trial))
        wanted.append(min(candidates, key=lambda v: v[0])[1])
    assert np.max(np.abs(got-wanted)) <= 5e-12
    # Compare ZF conditional phase likelihood with raw baseband Gaussian density.
    r = rng.normal(size=(32, 3, 2)); h = rng.normal(size=(32, 3, 2)); q = .1
    rc = r[..., 0]+1j*r[..., 1]; hc = h[..., 0]+1j*h[..., 1]; y = hc*rc
    theta = (grid.numpy()*2-1)*math.pi/2
    reference = -np.abs(y[..., None]-hc[..., None]*np.exp(1j*theta))**2/q
    observed = likelihood(torch.from_numpy(r), q, grid, torch.from_numpy(h)).numpy()
    reference -= reference.max(-1, keepdims=True); observed -= observed.max(-1, keepdims=True)
    assert np.max(np.abs(observed-reference)) <= 5e-10
    return dict(enumerated_observations=32, feasible_triples_per_observation=165,
        max_posterior_absolute_error=float(np.max(np.abs(actual-expected))), isotonic_cases=128,
        max_conditional_baseband_loglikelihood_error=float(np.max(np.abs(observed-reference))))


def integration_checks():
    spec = importlib.util.spec_from_file_location('frozen004', ROOT/'experiments/cost-field/code_004/codec.py')
    base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
    results = []
    for arm in ['G', 'P', 'S']:
        torch.manual_seed(17)
        codec = base.CostFieldCodec(torch.linspace(2, 59.6, 72), arm).eval()
        before = {k: digest(v) for k, v in codec.state_dict().items()}
        cost = torch.randn((1, 32, 72, 8, 12)); app = torch.randn((1, 32, 8, 12))
        prior = torch.randn((1, 1, 72, 8, 12)).softmax(2)
        with torch.inference_mode():
            packet, private = codec.encode(cost, app, prior)
            assert packet.shape == (1, 114, 2) and abs(float(packet.square().sum())-114) <= 64*np.finfo(np.float32).eps*114
            packet_hash = digest(packet)
            del cost, app, prior, private
            for channel in ['identity', 'awgn', 'rayleigh']:
                generator = torch.Generator().manual_seed(1715)
                received, physical = base.physical_channel(packet, channel, 10, generator)
                csi = physical['fading'] if channel == 'rayleigh' else None
                del physical
                reference = codec.decode(received, (8, 12), (8, 12))
                for mode in ['ordered', 'independent', 'isotonic', 'ordered_blind']:
                    result = decode(codec, received, (8, 12), (8, 12), channel, 10, csi, mode)
                    assert result[0].shape == (1, 32, 72, 8, 12) and result[1].shape == (1, 32, 8, 12)
                    assert all(bool(torch.isfinite(v).all()) for v in result)
                    assert torch.equal(result[1], reference[1])
                    if channel == 'identity':
                        assert all(torch.equal(a, b) for a, b in zip(result, reference))
                    repeat = decode(codec, received.clone(), (8, 12), (8, 12), channel, 10,
                        None if csi is None else csi.clone(), mode)
                    assert all(torch.equal(a, b) for a, b in zip(result, repeat))
                    results.append(dict(arm=arm, channel=channel, mode=mode,
                        full_output_sha256=[digest(v) for v in result], appearance_exact_frozen=True,
                        identity_exact_frozen=channel == 'identity'))
                assert digest(packet) == packet_hash and before == {k: digest(v) for k, v in codec.state_dict().items()}
        rejected = False
        try:
            decode(codec, received, (8, 12), (8, 12), 'rayleigh', 10, csi, clean_mu=torch.zeros(3))
        except TypeError:
            rejected = True
        assert rejected
    return dict(cases=results, private_clean_mu_rejected_each_arm=True,
        parameters_added=0, charged_complex_uses_per_pooled_site=19,
        source_and_model_states_unchanged=True, scope='synthetic8x12_feature_codec_no_dataset_detector_or_training')


def statistics(values):
    return dict(mean=float(np.mean(values)), standard_error=float(np.std(values, ddof=1)/math.sqrt(len(values))))


def risk_checks(artifact_directory):
    samples = 65536; batch = 4096; size = 33
    triples = np.asarray(list(itertools.combinations_with_replacement(range(size), 3)), dtype=np.int16)
    grid = torch.linspace(0, 1, size, dtype=torch.float64); gn = grid.numpy()
    field_grid = np.linspace(0, 1, 9); sigma = 1/3; coefficients = np.array([.8, -.3, .5])
    basis = np.exp(-.5*((field_grid[:, None]-gn[None, :])/sigma)**2)/3
    conditions = [('awgn', snr) for snr in range(6, 19)]+[('rayleigh', snr) for snr in range(6, 19, 2)]
    summaries = []
    with torch.inference_mode():
        for kind, snr in conditions:
            rng = np.random.default_rng(1715+1000*(kind == 'rayleigh')+snr)
            q = 10**(-snr/10)
            coordinates = {k: [] for k in ['hard', 'independent', 'ordered', 'isotonic', 'ordered_blind']}
            fields = {k: [] for k in [*coordinates, 'ordered_mean']}
            for offset in range(0, samples, batch):
                truth = gn[triples[rng.integers(0, len(triples), size=batch)]]
                tx = np.exp(1j*(truth*2-1)*math.pi/2)
                noise = rng.normal(size=(batch, 3, 2))*math.sqrt(q/2)
                h = np.ones((batch, 3), dtype=np.complex128)
                if kind == 'rayleigh':
                    rawh = rng.normal(size=(batch, 3, 2))/math.sqrt(2)
                    h = rawh[..., 0]+1j*rawh[..., 1]
                baseband = h*tx+noise[..., 0]+1j*noise[..., 1]
                received = baseband/h
                r = torch.from_numpy(np.stack((received.real, received.imag), -1))
                csi = None if kind == 'awgn' else torch.from_numpy(np.stack((h.real, h.imag), -1))
                ll = likelihood(r, q, grid, csi)
                posterior = {k: marginals(ll, ordered=k == 'ordered') for k in ['ordered', 'independent']}
                posterior['ordered_blind'] = (posterior['ordered'] if kind == 'awgn' else
                    marginals(fading_blind_likelihood(r, q, grid)))
                estimates = {k: (p@grid).numpy() for k, p in posterior.items()}
                estimates['hard'] = (np.clip(np.angle(received), -math.pi/2, math.pi/2)/(math.pi/2)+1)/2
                estimates['isotonic'] = isotonic_three(torch.from_numpy(estimates['hard']),
                    torch.from_numpy(np.abs(h)**2)).numpy()
                truth_field = (np.exp(-.5*((field_grid[None, None, :]-truth[..., None])/sigma)**2)/3*coefficients[None, :, None]).sum(1)
                for k, values in estimates.items():
                    coordinates[k].append(np.mean((values-truth)**2, axis=1))
                    if k in posterior:
                        predicted = np.einsum('nkg,dg,k->nd', posterior[k].numpy(), basis, coefficients)
                    else:
                        predicted = (np.exp(-.5*((field_grid[None, None, :]-values[..., None])/sigma)**2)/3*coefficients[None, :, None]).sum(1)
                    fields[k].append(np.mean((predicted-truth_field)**2, axis=1))
                values = estimates['ordered']
                predicted = (np.exp(-.5*((field_grid[None, None, :]-values[..., None])/sigma)**2)/3*coefficients[None, :, None]).sum(1)
                fields['ordered_mean'].append(np.mean((predicted-truth_field)**2, axis=1))
            coordinates = {k: np.concatenate(v) for k, v in coordinates.items()}
            fields = {k: np.concatenate(v) for k, v in fields.items()}
            artifact = artifact_directory/f'{kind}-snr{snr}.npz'
            assert not artifact.exists()
            np.savez_compressed(artifact, **{'coordinate_'+k: v for k, v in coordinates.items()},
                **{'field_'+k: v for k, v in fields.items()})
            artifact_identity = dict(path=str(artifact), sha256=sha(artifact),
                paired_trials=samples, named_array_count=len(coordinates)+len(fields))
            comparisons = {}
            for family, values in [('coordinate', coordinates), ('field', fields)]:
                for k in values:
                    if k == 'ordered':
                        continue
                    delta = values['ordered']-values[k]; stat = statistics(delta)
                    assert stat['mean'] <= 8*stat['standard_error']+1e-12, (kind, snr, family, k, stat)
                    comparisons[family+'_ordered_minus_'+k] = stat
            summary = dict(channel=kind, SNR_dB=snr, samples=samples, paired_risk_artifact=artifact_identity,
                coordinate_unit_MSE={k: statistics(v) for k, v in coordinates.items()},
                coordinate_latent_m2_MSE={k: float(v.mean()*57.6**2) for k, v in coordinates.items()},
                fixed_coefficient_field_MSE={k: statistics(v) for k, v in fields.items()},
                paired_differences=comparisons)
            summaries.append(summary)
            print(json.dumps(dict(channel=kind, SNR_dB=snr, ordered_m2=summary['coordinate_latent_m2_MSE']['ordered'])), flush=True)
    return dict(prior='Uniform over each of6545 inclusive ordered triples on33-point grid',
        conditions=summaries, physical_triples=20*samples, physical_complex_position_uses=3*20*samples,
        source_images_or_actual_KITTI_prior_or_object_depths_used=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); assert not args.output.exists()
    report = dict(state='starting', started_unix=time.time(), actual_AP_or_new_theorem_claimed=False)
    try:
        lock = json.loads((HERE/'inputs.json').read_text())
        assert all(sha(ROOT/p) == v for p, v in lock['files'].items())
        torch.set_num_threads(2)
        report.update(inputs_sha256=sha(HERE/'inputs.json'), torch_version=torch.__version__, numpy_version=np.__version__)
        report['enumeration'] = exact_checks()
        report['receiver_integration'] = integration_checks()
        artifact_directory = args.output.with_suffix('.paired'); artifact_directory.mkdir()
        report['declared_prior_risk'] = risk_checks(artifact_directory)
        assert all(sha(ROOT/p) == v for p, v in lock['files'].items())
        report['state'] = 'passed_exact_grid_inference_received_only_integration_and_all20_retained_paired_risks'
    except BaseException:
        report.update(state='failed_retained', traceback=traceback.format_exc())
        raise
    finally:
        report['ended_unix'] = time.time()
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
