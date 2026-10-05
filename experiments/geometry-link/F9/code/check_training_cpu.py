"""F9 training-scope, target-barrier and four-arm noise pairing CPU gates."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(HERE), str(ROOT / 'src')]
import conditions as c
import audit as a
from coupling import ARMS, install
from native_observer import ScopedNativeTaskAdaptation, freeze_scope
from geocomm.stereo_feature_link import StereoFeatureLink
from geocomm.stereo_task_adaptation import assert_frozen
from geocomm.student import StereoStudentEncoder

spec = importlib.util.spec_from_file_location('target_barrier_fixture', ROOT / 'tests/test_stereo_task_adaptation.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
sensor_spec = importlib.util.spec_from_file_location('augmented_fixture', ROOT / 'experiments/geometry-link/F7/code/test_conditions.py')
sensor_fixture = importlib.util.module_from_spec(sensor_spec)
sensor_spec.loader.exec_module(sensor_fixture)


def model(arm):
    result = fixture.Model()
    result.backbone_3d.student_semantic_link_encoder = StereoStudentEncoder()
    result.backbone_3d.stereo_feature_link.channel.kind = 'awgn'
    install(result.backbone_3d.stereo_feature_link, arm)
    return result


def inputs():
    result = fixture.inputs()
    result.update(left_img=torch.randn(1, 3, 64, 128), right_img=torch.randn(1, 3, 64, 128), image_shape=[[64, 128]])
    return result


class Checks(unittest.TestCase):
    def test_all_four_actual_scopes_backward_Adam_and_frozen_state(self):
        for arm in ARMS:
            m = model(arm)
            selected = freeze_scope(m, arm)
            self.assertEqual(len(selected), 51 if arm == 'U' else 55)
            names = {n for n, _ in selected}
            reference = {n: v.clone() for n, v in m.state_dict().items()}
            optimizer = torch.optim.AdamW([dict(params=[p for _, p in selected], parameter_names=[n for n, _ in selected])], lr=.0001, weight_decay=.0001)
            observer = ScopedNativeTaskAdaptation(m, arm)
            try:
                for step, gt in enumerate((torch.ones(1, 1, 8), torch.empty(1, 0, 8))):
                    optimizer.zero_grad(set_to_none=True)
                    loss = observer.forward(inputs(), gt)
                    loss.backward()
                    for name, parameter in selected:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                        if step == 0 and 'gain_head.0.' in name:
                            self.assertEqual(int(torch.count_nonzero(parameter.grad)), 0)
                    self.assertTrue(all(p.grad is None for n, p in m.named_parameters() if n not in names))
                    torch.nn.utils.clip_grad_norm_([p for _, p in selected], 10., error_if_nonfinite=True)
                    optimizer.step()
                    row = observer.finish_backward()
                    self.assertTrue(row['GT_introduced_only_at_head'])
                    if step == 0:
                        self.assertGreater(row['sender_profile']['wall_seconds'], 0)
                        self.assertIsNone(row['sender_profile']['cuda_elapsed_ms'])
                        self.assertEqual(row['sender_profile']['correspondence_executed'], arm in ('P', 'S'))
                    else:
                        self.assertTrue(row['empty_GT'])
                self.assertEqual(len(optimizer.state), len(selected))
                self.assertTrue(all(float(s['step']) == 2 for s in optimizer.state.values()))
                saved = dict(model_state=m.state_dict(), optimizer_state=optimizer.state_dict())
                a.audit_optimizer(saved, [n for n, _ in selected], 2)
                corrupted = copy.deepcopy(saved)
                next(iter(corrupted['optimizer_state']['state'].values()))['step'] = torch.tensor(1.)
                with self.assertRaises(ValueError): a.audit_optimizer(corrupted, [n for n, _ in selected], 2)
                self.assertEqual(assert_frozen(reference, m.state_dict(), names), len(reference) - len(selected))
                self.assertEqual(observer.calls['forbidden'], 0)
            finally:
                observer.close()

    def test_GT_and_forbidden_sources_cannot_reach_sender(self):
        m = model('P'); freeze_scope(m, 'P')
        observer = ScopedNativeTaskAdaptation(m, 'P')
        batch = inputs(); received = []
        hook = m.backbone_3d.stereo_feature_link.register_forward_hook(lambda module, i, out: received.append(tuple(v.detach().clone() for v in out)))
        rng = torch.get_rng_state().clone()
        try:
            for value in (1., 7.):
                torch.set_rng_state(rng); m.zero_grad(set_to_none=True)
                observer.forward(batch, torch.full((1, 1, 8), value)).backward()
                observer.finish_backward()
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(*received)))
            with self.assertRaises(RuntimeError): m.lidar_model({})
            with self.assertRaises(ValueError): observer.forward(dict(batch, gt_boxes=torch.ones(1, 1, 8)), torch.ones(1, 1, 8))
        finally:
            hook.remove(); observer.close()

    def test_scope_rejects_extra_parameters_wrong_arm_and_detached_student(self):
        m = model('G')
        with self.assertRaises(ValueError): freeze_scope(m, 'P')
        m.backbone_3d.stereo_feature_link.register_parameter('extra', torch.nn.Parameter(torch.ones(())))
        with self.assertRaises(ValueError): freeze_scope(m, 'G')
        m = model('S'); freeze_scope(m, 'S'); observer = ScopedNativeTaskAdaptation(m, 'S')
        try:
            with self.assertRaises(RuntimeError): observer.student(m.backbone_3d.student_semantic_link_encoder, (), (torch.ones(1), torch.ones(1)))
        finally:
            observer.close()

    def test_fixed_new_seeds_noise_replay_and_complete_four_arm_pairing(self):
        import eligibility
        self.assertEqual(eligibility.ROOT, ROOT)
        self.assertEqual(c.PROTOCOL_SHA, hashlib.sha256((ROOT / 'experiments/geometry-link/F9-epipolar-conditioned-JSCC.md').read_bytes()).hexdigest())
        self.assertEqual((c.SNR_SEED, c.NOISE_SEED), (1927, 1928))
        paths = {}; all_rows = {}; noise = {}
        with tempfile.TemporaryDirectory() as directory:
            for arm in ARMS:
                controller = c.PairedChannel(StereoFeatureLink(channel='awgn').eval(), 'awgn', 'cpu')
                rows = []; noisy = []; state = torch.get_rng_state().clone()
                try:
                    for step in range(1, 7):
                        sensors, gt = sensor_fixture.sensors(frame=f'{step:06d}', empty=step == 6)
                        snr = controller.prepare(step)
                        out = controller.link.channel(torch.ones(1, 62400, 2), snr)
                        noisy.append(out[0].clone())
                        rows.append(dict(step=step, epoch=1, optimization_scope=arm, channel='awgn', snr_db=snr, frame_id=f'{step:06d}', GT_shape=list(gt.shape), augmented_evidence=c.augmented_evidence(sensors, gt), channel_condition=copy.deepcopy(controller.last_evidence)))
                    self.assertTrue(torch.equal(state, torch.get_rng_state()))
                    self.assertEqual(c.replay_rng_checkpoint(controller.checkpoint_rng(), 'awgn', 6, rows)['state'], 'passed')
                    with self.assertRaises(ValueError): controller.prepare(6)
                finally:
                    controller.close()
                noise[arm] = noisy; all_rows[arm] = rows; paths[arm] = Path(directory) / (arm + '.jsonl')
            for arm in ARMS:
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(noise['U'], noise[arm])))
            def save():
                for arm in ARMS: paths[arm].write_text(''.join(json.dumps(row) + '\n' for row in all_rows[arm]))
            save(); self.assertEqual(c.paired_records(paths, 6)['state'], 'passed')
            with self.assertRaises(ValueError): c.paired_records({k: v for k, v in paths.items() if k != 'S'}, 6)
            original = copy.deepcopy(all_rows['S'][0])
            all_rows['S'][0]['channel_condition']['noise_rng_after_sha256'] = '0' * 64; save()
            with self.assertRaises(ValueError): c.paired_records(paths, 6)
            all_rows['S'][0] = original
            all_rows['S'][0]['augmented_evidence']['arrays']['left_img']['sha256'] = '0' * 64; save()
            with self.assertRaises(ValueError): c.paired_records(paths, 6)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and not torch.cuda.is_available()
    torch.set_num_threads(2); torch.manual_seed(17)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    report = dict(state='passed' if result.wasSuccessful() else 'failed', tests=result.testsRun,
                  failures=[str(t) + '\n' + trace for t, trace in result.failures], errors=[str(t) + '\n' + trace for t, trace in result.errors],
                  torch=torch.__version__, CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
                  scope='CPU target barrier, actual selected Adam updates and four-arm noise/data pairing; no native KITTI/AP')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    if not result.wasSuccessful(): raise SystemExit(1)


if __name__ == '__main__':
    main()
