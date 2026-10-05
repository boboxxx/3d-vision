"""Whole transferred F9 records/endpoint verification, no model or AP rerun."""
from contextlib import ExitStack
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'stereo-epipolar-native-seed17-001'
ARMS = ('U', 'G', 'P', 'S')
sys.path.insert(0, str(ROOT / 'experiments/geometry-link/F9/evaluation'))
from record_contract import validate as validate_inference_coding


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1048576), b''): h.update(b)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())
def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
def mapped(path):
    path = Path(path)
    if path.is_relative_to('/home/sheng/paper6'): return ROOT / path.relative_to('/home/sheng/paper6')
    assert path.is_relative_to('/mnt/d/paper6/runs')
    return ROOT / 'data/engineering/F9-formal-native-transfer-001' / str(path).lstrip('/')


def main():
    output = ROOT / 'data/provenance' / (PREFIX + '-local-verification-001.json')
    assert not output.exists()
    closure_path = ROOT / 'data/provenance' / (PREFIX + '-closure.json')
    closure = read(closure_path)
    assert closure['state'] == 'closed_all_four_training_and16_final_endpoints_actual_terminal'
    assert closure['actual_terminal'] and closure['ps_returncode'] == 1 and len(closure['actual_ps'].splitlines()) <= 1
    assert not closure['engineering'] and closure['original21_unchanged']
    assert (closure['training_updates_per_arm'], closure['raw_training_rows'], closure['prediction_files']) == (3340, 13360, 5952)
    for p, digest in closure['artifacts_sha256'].items(): assert sha(ROOT / p) == digest
    assert {p: str(mapped(n).relative_to(ROOT)) for p,n in closure['native_to_local_artifacts'].items()} == {p:p for p in closure['artifacts_sha256']}
    cycle = read(ROOT / 'data/runs' / (PREFIX + '-cycle.json'))
    launch = read(ROOT / 'data/runs' / (PREFIX + '-launch.json'))
    assert closure['cycle_pid'] == cycle['pid'] == launch['pid']
    assert len(cycle['completed_commands']) == 56 and all(x['returncode'] == 0 for x in cycle['completed_commands'])
    roots = {k: ROOT for k in ('project', 'experiment', 'F7', 'F8', 'evaluation')}
    roots.update(liga=ROOT / 'third_party/LIGA-Stereo', mmdet=ROOT / 'third_party/mmdetection_kitti', stereo_rcnn=ROOT / 'third_party/Stereo-RCNN')
    source = read(ROOT / 'data/provenance' / ('server-source-manifest-' + PREFIX + '.json'))
    identities = {k: v['actual_sources'] for k, v in source.items()}
    identities['evaluation'] = read(ROOT / 'data/provenance' / ('server-evaluation-source-manifest-' + PREFIX + '.json'))['evaluation']
    assert set(identities) == set(roots) and identities == cycle['source_identities']
    for key, item in identities.items():
        files = {p: sha(roots[key] / p) for p in item['file_hashes']}
        assert files == item['file_hashes'] and hashlib.sha256(canonical(files)).hexdigest() == item['sha256']
    for p, digest in {**cycle['protocols_sha256'], **cycle['CPU_gates_sha256']}.items(): assert sha(ROOT / p) == digest
    original = read(ROOT / 'data/provenance/cao2025-staged-trainer-source-bfee9d8.json')
    assert len(original) == 21 and all(sha(ROOT / 'reproduction/cao2025' / p) == digest for p, digest in original.items())
    fold = read(ROOT / 'data/internal-tuning-fold-001.json')
    train_ids = set(fold['folds']['geocomm_tune_train']['ids'])
    holdout_ids = fold['folds']['geocomm_tune_holdout']['ids']
    assert len(train_ids) == 3340 and len(holdout_ids) == 372 and train_ids.isdisjoint(holdout_ids)
    pair_path = ROOT / 'data/runs' / (PREFIX + '-pair-audit.json')
    pair = read(pair_path)
    assert pair['state'] == 'passed' and pair['steps'] == 3340 and pair['arms'] == list(ARMS)
    assert sha(pair_path) == closure['paired_audit_sha256'] == cycle['paired_audit_sha256']
    runs = {}; raw_paths = {}; empty = dict.fromkeys(ARMS, 0); prior = dict.fromkeys(ARMS)
    for arm in ARMS:
        item = cycle['training'][arm]
        run, audit = read(ROOT / item['manifest_path']), read(ROOT / item['audit_path'])
        assert run['state'] == 'finished' and audit['state'] == 'passed' and run['arm'] == audit['arm'] == arm
        assert run['optimizer_steps'] == audit['steps'] == audit['all_optimizer_update_counts'] == 3340
        assert run['full_state_tensors'] == 539 and run['initialization_sha256'] == '6e54ebc5fa696db26b3902cdd129ff42544635db4cfb56dfdb7f4b94841c5193'
        assert audit['inactive_states_identical'] == (488 if arm == 'U' else 484)
        assert audit['trained_parameter_tensors'] == (51 if arm == 'U' else 55)
        assert run['checkpoint_sha256'] == audit['checkpoint_sha256'] == closure['training'][arm]['final_checkpoint_sha256']
        assert audit['noise_rng_replay']['state'] == 'passed' and audit['noise_rng_replay']['actual_calls'] == 3340
        raw = (ROOT / item['audit_path']).with_suffix('.records') / 'training.jsonl'
        assert sha(raw) == run['training_records_sha256'] == audit['raw_snapshot_sha256']
        runs[arm] = run; raw_paths[arm] = raw
    for key in ('full_initialization_sha256', 'initialization_sha256', 'dataset', 'schedule', 'config_sha256', 'protocol_sha256'):
        assert all(runs[a][key] == runs['U'][key] for a in ARMS)
    rng = random.Random(1927); fingerprints = []; ids = []; steps = 0
    with ExitStack() as stack:
        streams = [stack.enter_context(raw_paths[a].open()) for a in ARMS]
        for steps, lines in enumerate(itertools.zip_longest(*streams), 1):
            assert steps <= 3340 and all(line is not None and line.endswith('\n') for line in lines)
            rows = {a: json.loads(line) for a, line in zip(ARMS, lines)}
            baseline = rows['U']; snr = rng.uniform(0, 20)
            ids.append(baseline['frame_id']); fingerprints.append(baseline['augmented_evidence']['sha256'])
            for arm, row in rows.items():
                assert row['step'] == steps and row['epoch'] == 1 and row['optimization_scope'] == arm
                assert row['frame_id'] == baseline['frame_id'] and row['augmented_evidence'] == baseline['augmented_evidence']
                assert row['channel_condition'] == baseline['channel_condition'] and row['snr_db'] == snr and row['channel'] == 'awgn'
                fp = row['augmented_evidence']; payload = {k: v for k, v in fp.items() if k != 'sha256'}
                assert hashlib.sha256(canonical(payload)).hexdigest() == fp['sha256'] and fp['frame_id'] == row['frame_id']
                cond = row['channel_condition']
                assert cond['step'] == steps and cond['snr_db'] == snr and cond['noise_seed'] == 1928 and cond['isolated_noise_generator']
                if prior[arm] is not None: assert cond['noise_rng_before_sha256'] == prior[arm]
                assert cond['noise_rng_before_sha256'] != cond['noise_rng_after_sha256']; prior[arm] = cond['noise_rng_after_sha256']
                assert row['GT_introduced_only_at_head'] and row['all_modules_eval'] and row['cost_inputs_are_received_features'] and row['appearance_input_is_received_feature']
                assert (row['total_complex_uses'], row['stereo_complex_uses'], row['appearance_complex_uses'], row['pilot_complex_uses'], row['header_complex_uses']) == (62400, 49920, 12480, 0, 0)
                assert abs(row['tx_energy'] / 62400 - 1) <= 1e-4
                assert all(math.isfinite(row[k]) and row[k] >= 0 for k in ('loss', 'classification_loss', 'location_loss', 'direction_loss', 'iou_loss_raw', 'preclip_grad_norm'))
                expected_loss = sum(row[k] for k in ('classification_loss', 'location_loss', 'direction_loss')) + row['iou_weight'] * row['iou_loss_raw']
                assert math.isclose(row['loss'], expected_loss, rel_tol=3e-6, abs_tol=1e-6)
                assert set(row['parameter_gradients']) == set(runs[arm]['trainable_parameter_names'])
                assert all(g['finite'] and math.isfinite(g['norm']) and g['norm'] >= 0 for g in row['parameter_gradients'].values())
                coding = row['F9_coding']; energy = coding['stereo_group_energy']; gains = coding['amplitude_gains']
                assert coding['arm'] == arm and coding['nominal_snr'] == snr and not coding['receiver_side_information']
                assert len(energy) == len(gains) == 64 and all(math.isfinite(v) and v >= 0 for v in energy)
                assert all(math.isfinite(v) and .5 <= v <= 2 for v in gains)
                assert math.isclose(sum(energy) + coding['appearance_energy'], coding['actual_energy_float64'], rel_tol=1e-12, abs_tol=1e-8)
                assert abs(coding['actual_energy_float64'] / 62400 - 1) <= 1e-5
                if arm == 'U' or steps == 1: assert gains == [1.] * 64
                empty[arm] += int(row['empty_GT'])
    assert steps == 3340 and len(set(ids)) == 3340 and set(ids) == train_ids
    assert all(n == 62 for n in empty.values())
    assert pair['ordered_fingerprints_sha256'] == hashlib.sha256(canonical(fingerprints)).hexdigest()
    assert pair['final_noise_rng_sha256'] == prior
    endpoints = {f'{a}-{c}{s}' for a in ARMS for c, s in (('identity', 10), ('awgn', 6), ('awgn', 10), ('awgn', 18))}
    assert set(closure['evaluations']) == set(cycle['evaluations']) == endpoints
    metrics = {}; predictions = 0; inference_rows = 0
    for label, item in closure['evaluations'].items():
        arm, channel, snr = item['arm'], item['channel'], item['snr_db']
        rid = f'{PREFIX}-{arm}-test-{channel}{snr}'
        run = read(ROOT / 'data/runs' / (rid + '.json'))
        ap = read(ROOT / 'data/runs' / (rid + '-AP-audit.json'))
        feature = read(ROOT / 'data/runs' / (rid + '-feature-audit.json'))
        boundary = read(ROOT / 'data/runs' / (rid + '-boundary-report.json'))
        assert run['state'] == boundary['state'] == 'finished' and ap['state'] == feature['state'] == 'passed'
        assert ap['frames'] == feature['frames'] == boundary['frames'] == 372 and feature['readonly_states'] == 539
        assert len(boundary['final_state_hashes']) == 539 and boundary['final_state_hashes'] == boundary['checkpoint_loading']['state_hashes']
        assert boundary['checkpoint_loading']['role'] == 'formal_final'
        assert run['run_checkpoint_sha256'] == item['checkpoint_sha256'] == runs[arm]['checkpoint_sha256'] == boundary['checkpoint_sha256']
        assert ap['F9_arm'] == feature['arm'] == boundary['arm'] == arm and ap['F9_snr_db'] == feature['snr_db'] == boundary['snr_db'] == snr
        assert ap['evaluation_sources'] == feature['evaluation_sources'] == boundary['evaluation_sources'] == identities['evaluation']
        directory = mapped(run['metrics']['path']).parent
        assert set(ap['files']) == set(holdout_ids)
        for frame in holdout_ids:
            assert sha(directory / 'final_result/data' / (frame + '.txt')) == ap['files'][frame]['prediction_sha256']; predictions += 1
        records = [json.loads(line) for line in (ROOT / 'data/runs' / (rid + '-features.jsonl')).read_text().splitlines()]
        assert [row['frame_id'] for row in records] == holdout_ids
        noise_end = None
        for row in records:
            validate_inference_coding(row, arm, channel, snr)
            if noise_end is not None: assert row['noise_rng_before'] == noise_end
            noise_end = row['noise_rng_after']; inference_rows += 1
        value = {k: v for k, v in ap['recomputed_metrics'].items() if k.startswith('Car_3d/')}
        assert value == item['Car3D_AP_R40_percent']; metrics[label] = value
    assert predictions == inference_rows == 5952
    differences = {f'P_minus_{a}': metrics['P-awgn10']['Car_3d/moderate_R40'] - metrics[a + '-awgn10']['Car_3d/moderate_R40'] for a in ('G', 'S', 'U')}
    assert differences == closure['prelocked_AWGN10_Moderate_differences_pp'] == cycle['prelocked_AWGN10_Moderate_differences_pp']
    result = dict(state='passed_all_transferred_artifacts13360_training_rows16_final_endpoints', checked_unix=time.time(),
        closure_sha256=sha(closure_path), verifier_sha256=sha(Path(__file__)), artifacts_verified=len(closure['artifacts_sha256']),
        training_rows=13360, paired_steps=3340, empty_GT=empty, inference_rows=5952, prediction_files=5952,
        metrics=metrics, prelocked_AWGN10_Moderate_differences_pp=differences,
        scope='Local complete metadata/raw-record/prediction-file checks; saved checkpoint, CUDA RNG and official AP replay are server audits')
    with output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(state=result['state'], differences=differences)), flush=True)


if __name__ == '__main__': main()
