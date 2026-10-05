"""F9 evaluation full-state/checkpoint-role/receiver and RNG CPU regressions."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import types

import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(HERE), str(ROOT / 'src')]
import common as c
from geocomm.student import StereoStudentEncoder
from geocomm.pooling_diagnostic import state_hashes
from receiver_reference import ReceiverReference

spec = importlib.util.spec_from_file_location('evaluation_boundary_fixture', ROOT / 'tests/test_stereo_task_adaptation.py')
fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)


def model(arm='P', channel='identity', snr=10):
    m = fixture.Model()
    m.backbone_3d.student_semantic_link_encoder = StereoStudentEncoder()
    m.backbone_3d.stereo_feature_link.channel.kind = channel
    m.backbone_3d.stereo_feature_link.snr_db = float(snr)
    # Synthetic fixed detector states make the exact native count testable on CPU.
    for index in range(482): m.register_buffer('fixed_fixture_' + str(index), torch.tensor(float(index)))
    m.eval(); c.treatment(m, arm, channel, snr)
    assert len(m.state_dict()) == 539
    return m


def checkpoint(m):
    return dict(model_state={k: v.clone() for k, v in m.state_dict().items()}, epoch=1, it=3340,
                arm=m.backbone_3d.stereo_feature_link.arm, version='F9-matched-native3D-codec-adaptation')


def sensors():
    batch = fixture.inputs()
    batch.update(left_img=torch.randn(1, 3, 64, 128), right_img=torch.randn(1, 3, 64, 128), image_shape=[[64, 128]])
    return batch


class Checks(unittest.TestCase):
    def test_independent_F9_record_contract_and_public_SNR_corruptions(self):
        from record_contract import validate
        for arm in ('U', 'G', 'P', 'S'):
            for channel in ('identity', 'awgn'):
                path = ROOT / 'data/runs/stereo-epipolar-evaluation-integration-001-raw' / f'{arm}-{channel}10' / 'records.jsonl'
                row = json.loads(path.read_text().splitlines()[0])
                validate(row, arm, channel, 10)
                if channel == 'awgn':
                    for snr in (6, 18):
                        example = copy.deepcopy(row)
                        example['accounting']['snr_db'] = example['F9_coding']['nominal_snr'] = snr
                        validate(example, arm, channel, snr)
                for owner, key, value in [('accounting', 'header_complex_uses', 1),
                    ('accounting', 'snr_db', 11), ('accounting', 'total_complex_uses', 62401),
                    ('F9_coding', 'receiver_side_information', True), ('F9_coding', 'head_parameters', 560),
                    ('F9_coding', 'amplitude_gains', [.49] * 64), ('F9_coding', 'actual_energy_float64', 1.),
                    (None, 'sensor_input_keys', row['sensor_input_keys'] + ['gt_boxes']),
                    (None, 'autograd_enabled', True), (None, 'cost_inputs_are_received_features', False)]:
                    bad = copy.deepcopy(row)
                    (bad[owner] if owner else bad)[key] = value
                    with self.assertRaises(AssertionError): validate(bad, arm, channel, 10)

    def test_AP_audit_preserves_native_writer_GT_and_official_evaluation(self):
        original = ast.parse((ROOT / 'scripts/audit_liga_run.py').read_text())
        adapted = ast.parse((HERE / 'AP_audit.py').read_text())
        def named(tree, name): return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        self.assertEqual(ast.dump(named(original, 'audit_prediction')), ast.dump(named(adapted, 'audit_prediction')))
        def GT_loop(tree):
            return next(n for n in ast.walk(named(tree, 'main')) if isinstance(n, ast.For) and
                        isinstance(n.target, ast.Tuple) and [v.id for v in n.target.elts] == ['frame', 'anno', 'info'])
        self.assertEqual(ast.dump(GT_loop(original)), ast.dump(GT_loop(adapted)))
        def official_call(tree):
            return next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'get_official_eval_result')
        self.assertEqual(ast.dump(official_call(original)), ast.dump(official_call(adapted)))
        import AP_audit
        import numpy as np
        anno = dict(name=np.asarray(['Car']), alpha=np.asarray([.1]), bbox=np.asarray([[1., 2., 3., 4.]]),
                    dimensions=np.asarray([[3.9, 1.5, 1.6]]), location=np.asarray([[1., 2., 20.]]),
                    rotation_y=np.asarray([.2]), score=np.asarray([.8]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'prediction.txt'
            path.write_text('Car -1 -1 0.1 1 2 3 4 1.5 1.6 3.9 1 2 20 0.2 0.8\n')
            AP_audit.audit_prediction(anno, path)
            path.write_text('Car -1 -1 0.1 1 2 3 4 1.5 1.6 3.9 1 2 20 0.2 0.81\n')
            with self.assertRaises(AssertionError): AP_audit.audit_prediction(anno, path)

    def test_integration_sources_locked_ids_and_independent_prediction_audit(self):
        for path in HERE.glob('*.py'): ast.parse(path.read_text(), filename=str(path))
        import integration
        import integration_audit
        import formal_cycle
        closure_path = ROOT / 'data/provenance/stereo-epipolar-native-sanity-001-closure.json'
        closure = json.loads(closure_path.read_text())
        local = json.loads((ROOT / 'data/provenance/stereo-epipolar-native-sanity-001-local-verification-001.json').read_text())
        self.assertEqual(formal_cycle.training_engineering_gate(closure, local, c.sha256(closure_path))['records'], 24)
        with self.assertRaises(ValueError): formal_cycle.training_engineering_gate(closure, dict(local, records=23), c.sha256(closure_path))
        self.assertEqual(integration.IDS, ['000036', '000054', '000071', '000082', '000113', '000141'])
        self.assertEqual(integration.IDS, integration_audit.IDS)
        self.assertEqual(set(integration.source_specs()), {'project', 'liga', 'mmdet', 'stereo_rcnn', 'F7', 'F8', 'experiment', 'evaluation'})
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'paired.npz'
            values = dict(pred_boxes=np.ones((2, 7), np.float32), pred_scores=np.ones(2, np.float32), pred_labels=np.ones(2, np.int64))
            arrays = {prefix + k: v.copy() for prefix in ('explicit_', 'reference_') for k, v in values.items()}
            np.savez_compressed(path, **arrays); integration_audit.archive(path, paired=True)
            arrays['reference_pred_boxes'][0, 0] += 1
            np.savez_compressed(path, **arrays)
            with self.assertRaises(AssertionError): integration_audit.archive(path, paired=True)

    def reference_model(self):
        m = model('G')
        def head(self, batch):
            batch['fixture_boxes'] = batch['spatial_features_2d'].mean().expand(1, 7).clone()
            return batch
        def auxiliary(self, batch):
            batch['head_outs'] = [torch.ones(1)]
            batch['boxes_2d_pred'] = [dict(pred_boxes_2d=torch.ones(1, 4))]
            return batch
        m.dense_head.forward = types.MethodType(head, m.dense_head)
        m.dense_head_2d.forward = types.MethodType(auxiliary, m.dense_head_2d)
        m.post_processing = lambda batch: ([dict(pred_boxes=batch['fixture_boxes'], pred_scores=torch.ones(1),
                                                   pred_labels=torch.ones(1, dtype=torch.long))], {})
        c.install_native3D_forward(m)
        return m

    def test_isolated_same_received_reference_exact_and_once(self):
        m = self.reference_model(); rows = []
        observer = c.F9Observer(m, 'G', 'identity', 10, rows.append)
        reference = ReceiverReference(m, observer); before = state_hashes(m)
        try:
            with torch.no_grad():
                predictions, _ = m(sensors())
                report, arrays = reference.compare(predictions)
                self.assertTrue(report['all_boxes_scores_classes_exact'])
                self.assertEqual(observer.calls, dict(frames=1, student=2, codec=1, channel=1, build_cost=1, forbidden=0))
                self.assertEqual(observer.reference_calls, dict(dense_head_2d=1, depth_loss_head=1))
                self.assertTrue(all(torch.equal(torch.from_numpy(arrays['explicit_' + k]), torch.from_numpy(arrays['reference_' + k]))
                                    for k in ('pred_boxes', 'pred_scores', 'pred_labels')))
                with self.assertRaises(ValueError):
                    with observer.reference_scope(): pass
                with self.assertRaises(RuntimeError): m.depth_loss_head({})
            self.assertEqual(before, state_hashes(m))
        finally: reference.close(); observer.close()

    def test_reference_mutation_sender_teacher_and_label_access_rejected(self):
        for attack in ('mutation', 'sender', 'teacher', 'GT'):
            m = self.reference_model(); observer = c.F9Observer(m, 'G', 'identity', 10, lambda row: None)
            reference = ReceiverReference(m, observer)
            try:
                with torch.no_grad():
                    predictions, _ = m(sensors())
                    if attack == 'mutation':
                        original = m.dense_head_2d.forward
                        def mutated(batch):
                            batch['spatial_features_2d'].add_(1)
                            return original(batch)
                        m.dense_head_2d.forward = mutated
                        with self.assertRaises(ValueError): reference.compare(predictions)
                    else:
                        with self.assertRaises((ValueError, RuntimeError)):
                            with observer.reference_scope():
                                if attack == 'sender': m.backbone_3d(sensors())
                                elif attack == 'teacher': m.lidar_model({})
                                else: m.dense_head_2d(dict(gt_boxes=torch.ones(1)))
                    self.assertFalse(observer.in_reference)
            finally: reference.close(); observer.close()

    def test_explicit_native3D_forward_no_auxiliary_GT_or_training(self):
        m = model('G'); rows = []
        def native_head(self, batch):
            value = batch['spatial_features_2d'].mean()
            batch['fixture_boxes'] = value.expand(1, 7).clone()
            return batch
        def post_processing(batch):
            return [dict(pred_boxes=batch['fixture_boxes'], pred_scores=torch.ones(1), pred_labels=torch.ones(1, dtype=torch.long))], {}
        m.dense_head.forward = types.MethodType(native_head, m.dense_head)
        m.post_processing = post_processing
        observer = c.F9Observer(m, 'G', 'identity', 10, rows.append)
        before = state_hashes(m); c.install_native3D_forward(m)
        try:
            with torch.no_grad(): predictions, _ = m(sensors())
            self.assertEqual(len(rows), 1)
            self.assertIn('communication_accounting', predictions[0]['batch_dict'])
            self.assertEqual(before, state_hashes(m))
            self.assertEqual(observer.calls['forbidden'], 0)
            with self.assertRaises(ValueError): m(sensors())
            with torch.no_grad():
                with self.assertRaises(ValueError): m(dict(sensors(), gt_boxes=torch.ones(1, 1, 8)))
            with self.assertRaises(ValueError): c.install_native3D_forward(m)
        finally: observer.close()

    def test_exact539_checkpoint_loading_and_native_loader_preservation(self):
        source, target = model(), model()
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'full.pth'; torch.save(checkpoint(source), p)
            evidence = c.frozen_load(target, p, c.sha256(p))
            self.assertEqual(evidence['required_states'], 539)
            self.assertTrue(all(torch.equal(v, target.state_dict()[k]) for k, v in source.state_dict().items()))
            def broken_loader(filename, **kwargs): pass
            target.backbone_3d.stereo_feature_link.gain_head[2].bias.data.fill_(3.)
            with self.assertRaises(ValueError): c.frozen_load(target, p, c.sha256(p), loader=broken_loader)

    def test_wrong_count_shape_dtype_nonfinite_role_or_arm_rejected(self):
        m = model(); initial = checkpoint(m)
        corruptions = []
        value = copy.deepcopy(initial); value['model_state'].pop(next(iter(value['model_state']))); corruptions.append(value)
        value = copy.deepcopy(initial); value['model_state']['extra'] = torch.ones(()); corruptions.append(value)
        name = 'backbone_3d.stereo_feature_link.gain_head.2.weight'
        for tensor in (torch.zeros(2, 16), torch.zeros(1, 16, dtype=torch.float64), torch.full((1, 16), float('nan'))):
            value = copy.deepcopy(initial); value['model_state'][name] = tensor; corruptions.append(value)
        for key, new in [('arm', 'G'), ('it', 6), ('epoch', 0), ('version', 'engineering')]:
            value = copy.deepcopy(initial); value[key] = new; corruptions.append(value)
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'invalid.pth'
            for value in corruptions:
                torch.save(value, p)
                with self.assertRaises(ValueError): c.frozen_load(m, p, c.sha256(p))
            torch.save(initial, p)
            with self.assertRaises(ValueError): c.frozen_load(m, p, '0' * 64)
            with self.assertRaises(ValueError): c.frozen_load(m, p, c.sha256(p), role='fallback')

    def test_initialization_snapshot_separate_from_formal_final(self):
        m = model('S')
        value = dict(model_state=m.state_dict(), seed=17, version='F9-exact-F8joint-plus-four-head-states')
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'initialization.pth'; torch.save(value, p)
            self.assertEqual(c.frozen_load(m, p, c.sha256(p), role='engineering_initialization')['role'], 'engineering_initialization')
            with self.assertRaises(ValueError): c.frozen_load(m, p, c.sha256(p))

    def test_public_arm_SNR_config_and_evaluation_source_identity(self):
        for arm in ('U', 'G', 'P', 'S'):
            for channel, snr in [('identity', 10), ('awgn', 6), ('awgn', 10), ('awgn', 18)]:
                self.assertEqual(model(arm, channel, snr).backbone_3d.stereo_feature_link.arm, arm)
        m = fixture.Model(); m.backbone_3d.stereo_feature_link.snr_db = 6
        with self.assertRaises(ValueError): c.treatment(m, 'P', 'identity', 6)
        with self.assertRaises(ValueError): c.treatment(m, 'invalid', 'awgn', 6)
        with self.assertRaises(ValueError): c.treatment(m, 'P', 'awgn', 10)
        self.assertEqual(c.evaluation_sources(), c.evaluation_sources())
        self.assertEqual(c.ROOT, ROOT)

    def test_four_concrete_configs_keep_native_data_receiver_and_thresholds(self):
        original = yaml.safe_load((ROOT / 'configs/diagnostic/stereo_feature_awgn_holdout.yaml').read_text())
        for channel, snr in [('identity', 10), ('awgn', 6), ('awgn', 10), ('awgn', 18)]:
            cfg = yaml.safe_load((HERE / 'configs' / f'{channel}{snr}-holdout.yaml').read_text())
            link = cfg['MODEL']['BACKBONE_3D']['STEREO_FEATURE_LINK']
            self.assertEqual((link['channel'], link['snr_db']), (channel, float(snr)))
            link['channel'] = original['MODEL']['BACKBONE_3D']['STEREO_FEATURE_LINK']['channel']
            link['snr_db'] = original['MODEL']['BACKBONE_3D']['STEREO_FEATURE_LINK']['snr_db']
            self.assertEqual(cfg, original)

    def test_receive_only_observer_noise_and_readonly_states(self):
        for arm in ('U', 'G', 'P', 'S'):
            for channel in ('identity', 'awgn'):
                m = model(arm, channel); rows = []; observer = c.F9Observer(m, arm, channel, 10, rows.append)
                before = state_hashes(m)
                try:
                    with torch.no_grad(): m.backbone_3d(sensors())
                    self.assertEqual(len(rows), 1)
                    row = rows[0]
                    self.assertEqual(row['arm'], arm)
                    self.assertEqual(row['F9_coding']['receiver_side_information'], False)
                    self.assertEqual(row['noise_rng_before'] == row['noise_rng_after'], channel == 'identity')
                    for key in ('noise_rng_before', 'noise_rng_after'):
                        self.assertEqual(hashlib.sha256(bytes.fromhex(row[key]['state_hex'])).hexdigest(), row[key]['sha256'])
                    self.assertEqual(before, state_hashes(m))
                    with self.assertRaises(RuntimeError): m.lidar_model({})
                    with self.assertRaises(RuntimeError): m.dense_head_2d({})
                    with torch.no_grad():
                        with self.assertRaises(RuntimeError): m.backbone_3d(dict(sensors(), gt_boxes=torch.ones(1, 1, 8)))
                finally: observer.close()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); assert not args.output.exists() and not torch.cuda.is_available()
    torch.set_num_threads(2); torch.manual_seed(17)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    value = dict(state='passed' if result.wasSuccessful() else 'failed', tests=result.testsRun,
                 failures=[str(t) + '\n' + trace for t, trace in result.failures], errors=[str(t) + '\n' + trace for t, trace in result.errors],
                 source_sha256={str(p.relative_to(ROOT)): c.sha256(p) for p in HERE.glob('*.py')}, evaluation_source_identity=c.evaluation_sources(), torch=torch.__version__,
                 scope='Synthetic CPU539-state fixture and actual sender/receiver observer; no native KITTI, GPU inference or AP')
    args.output.write_text(json.dumps(value, indent=2) + '\n')
    if not result.wasSuccessful(): raise SystemExit(1)


if __name__ == '__main__': main()
