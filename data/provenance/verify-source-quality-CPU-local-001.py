"""Both complete quality fixture reports and pinned source/operator identities."""
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    paths = [ROOT / 'data/engineering' / ('source-quality-' + host + '-CPU-001.json') for host in ('local', 'sheng')]
    reports = [json.loads(path.read_text()) for path in paths]
    source = reports[0]['sources']
    for path, digest in source.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    for report in reports:
        assert report['sources'] == source and len(source) == 3
        assert report['state'] == 'passed_six_independent_RGB_quality_CPU_families' and report['families'] == 6
        assert len(report['SSIM_literal_max_abs_errors']) == 4 and report['SSIM_values_each_fixture'] == 351
        assert max(report['SSIM_literal_max_abs_errors']) == report['maximum_SSIM_error'] < 2e-11
        assert report['empty_and_perfect_JSON_policy_verified'] and report['quantization_half_ties_and_overshoot_verified']
        pool = report['pooled_fixture']
        assert pool['views'] == pool['defined_views'] == pool['secondary_finite_view_psnr_count'] == 2
        assert pool['empty_views'] == pool['perfect_views'] == 0
        assert pool['channel_values'] == 702 and pool['pixel_centers'] == 234
        assert abs(pool['pooled_mse'] - .085) < 1e-15
        assert abs(pool['pooled_psnr_dB'] + 10 * math.log10(.085)) < 1e-12
    output = ROOT / 'data/provenance/source-quality-CPU-transferred-verification-001.json'; assert not output.exists()
    result = dict(state='passed_both_complete_independent_quality_CPU_reports', checked_unix=time.time(),
                  reports={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                  sources=source, families_each=6, independent_SSIM_values_each_host=1404,
                  maximum_SSIM_errors_each=[report['maximum_SSIM_error'] for report in reports],
                  limitation='Complete source/fixture metadata verification; independent literal window checks executed on each host. No KITTI quality or AP.')
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({key: result[key] for key in ('state', 'maximum_SSIM_errors_each')}))


if __name__ == '__main__':
    main()
