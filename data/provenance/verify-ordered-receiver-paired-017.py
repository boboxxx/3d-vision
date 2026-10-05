"""Re-read every saved error array and independently reduce all paired metrics."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = [('awgn', x) for x in range(6, 19)]+[('rayleigh', x) for x in range(6, 19, 2)]
COORDINATE = ('hard', 'independent', 'ordered', 'isotonic', 'ordered_blind')
FIELD = (*COORDINATE, 'ordered_mean')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def reduce(values):
    mean = math.fsum(float(x) for x in values)/len(values)
    variance = math.fsum((float(x)-mean)**2 for x in values)/(len(values)-1)
    return dict(mean=mean, standard_error=math.sqrt(variance/len(values)))


def equal(a, b):
    assert abs(a-b) <= 64*np.finfo(np.float64).eps*(1+abs(b)), (a, b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); assert not args.output.exists()
    lockpath = ROOT/'experiments/cost-field/ordered-receiver-paired-audit-inputs-017.json'
    lock = json.loads(lockpath.read_text())
    assert all(sha(ROOT/p) == v for p, v in lock['files'].items())
    report = json.loads(args.report.read_text())
    assert report['state'] == 'passed_exact_grid_inference_received_only_integration_and_all20_retained_paired_risks'
    assert report['inputs_sha256'] == sha(ROOT/'experiments/cost-field/ordered-receiver-code-016/inputs.json')
    risk = report['declared_prior_risk']; assert risk['physical_triples'] == 1310720 and risk['physical_complex_position_uses'] == 3932160
    conditions = risk['conditions']; assert [(c['channel'], c['SNR_dB']) for c in conditions] == CONDITIONS
    directory = args.report.resolve().with_suffix('.paired')
    assert directory.is_relative_to(ROOT) and len(list(directory.glob('*.npz'))) == 20
    facts = []; total = 0
    for condition in conditions:
        kind, snr = condition['channel'], condition['SNR_dB']
        expected = directory/f'{kind}-snr{snr}.npz'
        item = condition['paired_risk_artifact']
        path = Path(item['path']); path = path if path.is_absolute() else ROOT/path
        assert path.resolve() == expected and path.is_file()
        digest = sha(path); assert digest == item['sha256']
        assert item['paired_trials'] == condition['samples'] == 65536 and item['named_array_count'] == 11
        with np.load(path, allow_pickle=False) as saved:
            assert set(saved.files) == {f'coordinate_{x}' for x in COORDINATE}|{f'field_{x}' for x in FIELD}
            arrays = {k: saved[k].copy() for k in saved.files}
        assert all(v.dtype == np.float64 and v.shape == (65536,) and np.isfinite(v).all() and (v >= 0).all() for v in arrays.values())
        for family, names, report_key in [('coordinate', COORDINATE, 'coordinate_unit_MSE'),
                                        ('field', FIELD, 'fixed_coefficient_field_MSE')]:
            for name in names:
                metric = reduce(arrays[family+'_'+name])
                for key in metric:
                    equal(metric[key], condition[report_key][name][key])
                if family == 'coordinate':
                    equal(metric['mean']*57.6**2, condition['coordinate_latent_m2_MSE'][name])
                if name != 'ordered':
                    delta = arrays[family+'_ordered']-arrays[family+'_'+name]
                    paired = reduce(delta); key = family+'_ordered_minus_'+name
                    for k in paired:
                        equal(paired[k], condition['paired_differences'][key][k])
                    assert paired['mean'] <= 8*paired['standard_error']+1e-12
        assert sha(path) == digest
        total += 65536*11
        facts.append(dict(channel=kind, SNR_dB=snr, sha256=digest, finite_saved_values=65536*11))
    assert all(sha(ROOT/p) == v for p, v in lock['files'].items())
    result = dict(state='passed_all20_complete_saved_paired_risks_independent_fsum_reduction',
        report_sha256=sha(args.report), inputs_sha256=sha(lockpath), verifier_sha256=sha(Path(__file__)),
        files=facts, saved_values_checked=total, physical_triples=1310720,
        actual_object_depth_AP_KITTI_calibration_or_novelty_claimed=False, checked_unix=time.time())
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(state=result['state'], saved_values_checked=total)))


if __name__ == '__main__':
    main()
