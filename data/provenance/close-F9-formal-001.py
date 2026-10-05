"""Post-terminal full F9 saved artifact/source/AP-file closure; never repeats jobs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/geometry-link/F9/evaluation'))
from formal_cycle import ARMS, ENDPOINTS, source_specs
from common import source_identity
from conditions import paired_records

PREFIX = 'stereo-epipolar-native-seed17-001'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''): h.update(part)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def main():
    runs, provenance = ROOT / 'data/runs', ROOT / 'data/provenance'
    output = provenance / (PREFIX + '-closure.json'); transfer = provenance / (PREFIX + '-transfer-files-001.txt')
    assert not output.exists() and not transfer.exists()
    launch_path = runs / (PREFIX + '-launch.json'); cycle_path = runs / (PREFIX + '-cycle.json')
    launch, cycle = read(launch_path), read(cycle_path)
    ps = subprocess.run(['ps', '-p', str(launch['pid']), '-o', 'pid,stat,etime,args'], text=True, capture_output=True)
    assert ps.returncode == 1 and len(ps.stdout.splitlines()) <= 1 and launch['pid'] == cycle['pid']
    assert cycle['state'] == 'finished_all_four_training_pair_16_AP_feature_audits_pending_terminal_closure'
    assert not cycle['engineering'] and cycle['updates_per_arm'] == 3340 and len(cycle['evaluations']) == 16
    assert len(cycle['completed_commands']) == 56 and set(cycle['training']) == set(ARMS)
    assert {k: source_identity(*v) for k, v in source_specs().items()} == cycle['source_identities']
    source_path = provenance / ('server-source-manifest-' + PREFIX + '.json')
    eval_path = provenance / ('server-evaluation-source-manifest-' + PREFIX + '.json')
    assert sha(source_path) == cycle['source_manifest_sha256'] and sha(eval_path) == cycle['evaluation_manifest_sha256']
    assert read(eval_path)['evaluation'] == cycle['source_identities']['evaluation']
    for p, h in {**cycle['protocols_sha256'], **cycle['CPU_gates_sha256']}.items(): assert sha(ROOT / p) == h
    originals = read(provenance / 'cao2025-staged-trainer-source-bfee9d8.json')
    assert len(originals) == 21 and all(sha(ROOT / 'reproduction/cao2025' / p) == h for p, h in originals.items())
    assert sha(ROOT / 'reproduction/cao2025/formal-protocol.md') == '682fb774e09b9be7f68d09f8251dac6d4b99af3290f02660951d398f9e00eace'
    artifacts = {launch_path, cycle_path, source_path, eval_path, ROOT / 'logs' / (PREFIX + '-controller.log')}
    artifacts.update(ROOT / p for p in cycle['CPU_gates_sha256'])
    native_large = {}; training_rows = {}; training = {}
    for arm, item in cycle['training'].items():
        manifest, audit_path = ROOT / item['manifest_path'], ROOT / item['audit_path']
        assert sha(manifest) == item['manifest_sha256'] and sha(audit_path) == item['audit_sha256']
        run, audit = read(manifest), read(audit_path)
        assert run['state'] == 'finished' and audit['state'] == 'passed' and run['arm'] == audit['arm'] == arm
        assert run['optimizer_steps'] == audit['steps'] == audit['all_optimizer_update_counts'] == 3340
        assert audit['inactive_states_identical'] == (488 if arm == 'U' else 484)
        assert audit['trained_parameter_tensors'] == (51 if arm == 'U' else 55) and audit['summary']['empty_GT_frames'] >= 62
        assert audit['summary']['unique_returned_frames'] == 3340
        assert run['source_manifest_sha256'] == sha(source_path) and audit['manifest_sha256'] == sha(manifest)
        assert sha(run['checkpoint_path']) == run['checkpoint_sha256'] == audit['checkpoint_sha256']
        assert sha(run['full_initialization_path']) == run['full_initialization_sha256'] == audit['full_initialization_sha256']
        assert run['full_state_tensors'] == 539
        raw = Path(run['output_dir']) / 'training.jsonl'; retained = audit_path.with_suffix('.records') / 'training.jsonl'
        assert sha(raw) == sha(retained) == run['training_records_sha256'] == audit['raw_snapshot_sha256']
        training_rows[arm] = retained; artifacts.update((manifest, audit_path, retained))
        native_large[arm] = {str(Path(run[k])): sha(run[k]) for k in ('checkpoint_path', 'full_initialization_path')}
        training[arm] = dict(updates=3340, selected=audit['trained_parameter_tensors'], fixed=audit['inactive_states_identical'], empty_GT=audit['summary']['empty_GT_frames'],
                             final_checkpoint_sha256=run['checkpoint_sha256'], peak_reserved_GiB=run['peak_reserved_GiB'])
    pair_path = runs / (PREFIX + '-pair-audit.json')
    pair = paired_records(training_rows, 3340)
    assert pair == read(pair_path) and sha(pair_path) == cycle['paired_audit_sha256']; artifacts.add(pair_path)
    for command in cycle['completed_commands']:
        p = ROOT / command['log_path']; assert command['returncode'] == 0 and sha(p) == command['log_sha256']; artifacts.add(p)
    fold = read(ROOT / 'data/internal-tuning-fold-001.json'); ids = fold['folds']['geocomm_tune_holdout']['ids']; assert len(ids) == 372
    expected = {f'{a}-{c}{s}' for a in ARMS for c, s in ENDPOINTS}
    assert set(cycle['evaluations']) == expected
    AP_files = 0
    for label, item in cycle['evaluations'].items():
        arm, channel, snr = item['arm'], item['channel'], item['snr_db']
        rid = f'{PREFIX}-{arm}-test-{channel}{snr}'
        for p, h in item['artifacts_sha256'].items(): assert sha(ROOT / p) == h; artifacts.add(ROOT / p)
        run = read(runs / (rid + '.json')); ap = read(runs / (rid + '-AP-audit.json'))
        feature = read(runs / (rid + '-feature-audit.json')); boundary = read(runs / (rid + '-boundary-report.json'))
        assert run['state'] == boundary['state'] == 'finished' and ap['state'] == feature['state'] == 'passed'
        assert ap['frames'] == feature['frames'] == boundary['frames'] == 372 and feature['readonly_states'] == 539
        assert len(boundary['final_state_hashes']) == 539 and boundary['final_state_hashes'] == boundary['checkpoint_loading']['state_hashes']
        assert boundary['checkpoint_loading']['role'] == 'formal_final'
        assert feature['executed_calls'] == boundary['calls'] == dict(frames=372, student=744, codec=372, channel=372, build_cost=372, forbidden=0)
        assert boundary['arm'] == ap['F9_arm'] == feature['arm'] == arm and boundary['snr_db'] == ap['F9_snr_db'] == feature['snr_db'] == snr
        assert boundary['evaluation_sources'] == feature['evaluation_sources'] == ap['evaluation_sources'] == cycle['source_identities']['evaluation']
        assert run['run_checkpoint_sha256'] == feature['checkpoint_sha256'] == boundary['checkpoint_sha256'] == item['checkpoint_sha256'] == training[arm]['final_checkpoint_sha256']
        assert ap['communication']['channel'] == [channel] and ap['communication']['complex_uses'] == [62400]
        assert item['Car3D_AP_R40_percent'] == {k: v for k, v in ap['recomputed_metrics'].items() if k.startswith('Car_3d/')}
        assert [json.loads(line)['frame_id'] for line in (runs / (rid + '-features.jsonl')).read_text().splitlines()] == ids
        metrics = Path(run['metrics']['path']); directory = metrics.parent
        assert sha(metrics) == run['metrics']['sha256'] == ap['metrics_sha256']
        assert sha(directory / 'result.pkl') == ap['result_pickle_sha256']
        artifacts.update((metrics, directory / 'result.pkl', directory / 'communication_rank0.jsonl'))
        assert set(ap['files']) == set(ids)
        data = Path(run['dataset']['root'])
        for frame in ids:
            for p, key in ((directory / 'final_result/data' / (frame + '.txt'), 'prediction_sha256'),
                           (data / 'training/label_2' / (frame + '.txt'), 'label_sha256'),
                           (data / 'training/calib' / (frame + '.txt'), 'calibration_sha256')):
                assert sha(p) == ap['files'][frame][key]
                # Shared KITTI labels/calibration may live outside this checkout.
                # They are independently rehashed here, retained by native AP audit;
                # prediction files/pickles/records are included in local transfer.
                if key == 'prediction_sha256': artifacts.add(p); AP_files += 1
    differences = {f'P_minus_{a}': cycle['evaluations']['P-awgn10']['Car3D_AP_R40_percent']['Car_3d/moderate_R40'] -
                   cycle['evaluations'][a + '-awgn10']['Car3D_AP_R40_percent']['Car_3d/moderate_R40'] for a in ('G', 'S', 'U')}
    assert differences == cycle['prelocked_AWGN10_Moderate_differences_pp']
    assert AP_files == 5952
    result = dict(state='closed_all_four_training_and16_final_endpoints_actual_terminal', prefix=PREFIX, checked_unix=time.time(),
        actual_terminal=True, actual_ps=ps.stdout, ps_returncode=ps.returncode, cycle_pid=launch['pid'], engineering=False,
        training_updates_per_arm=3340, raw_training_rows=13360, training=training, prediction_files=5952,
        paired_audit_sha256=sha(pair_path), native_large_sha256=native_large,
        source_manifest_sha256=sha(source_path), evaluation_manifest_sha256=sha(eval_path), original21_unchanged=True,
        artifacts_sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted(artifacts)},
        evaluations=cycle['evaluations'], prelocked_AWGN10_Moderate_differences_pp=differences,
        limitations=cycle['limitations'])
    with output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    transfer.write_text('\n'.join([*result['artifacts_sha256'], str(output.relative_to(ROOT))]) + '\n')
    print(json.dumps(dict(state=result['state'], artifacts=len(artifacts), rows=13360, endpoints=16, predictions=5952)))


if __name__ == '__main__': main()
