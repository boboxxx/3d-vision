"""F9 independent dense reference and meaningful CPU-only coding checks."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'src'))
import coupling as c
from geocomm.channel import ComplexChannel
from geocomm.stereo_feature_link import StereoFeatureLink


def dense_reference(left, right):
    """Independent explicit pixel/candidate loops; no implementation pooling/group helpers."""
    arrays = []
    for feature in (left, right):
        value = feature.detach().numpy()[0].astype(np.float64)
        channels, h, w = value.shape
        pooled = np.empty((channels, h // 4, w // 4), dtype=np.float64)
        for y in range(h // 4):
            for x in range(w // 4):
                pooled[:, y, x] = value[:, 4*y:4*y+4, 4*x:4*x+4].mean((1, 2))
        norm = np.sqrt((pooled * pooled).sum(0))
        arrays.append(pooled / np.maximum(norm, 1e-6)[None])
    h, w = arrays[0].shape[-2:]
    result = []
    for reverse in (False, True):
        source, target = arrays[::-1] if reverse else arrays
        matrix = np.zeros((32, 32), dtype=np.float64)
        counts = np.zeros(32, dtype=np.int64)
        for y in range(h):
            for x in range(w):
                source_group = int((y + .5) * 4 / h) * 8 + int((x + .5) * 8 / w)
                candidates = [x + d if reverse else x - d for d in range(49)]
                candidates = [xx for xx in candidates if 0 <= xx < w]
                scores = np.array([np.dot(source[:, y, x], target[:, y, xx]) / .1 for xx in candidates])
                probability = np.exp(scores - scores.max()); probability /= probability.sum()
                for xx, p in zip(candidates, probability):
                    target_group = int((y + .5) * 4 / h) * 8 + int((xx + .5) * 8 / w)
                    matrix[source_group, target_group] += p
                counts[source_group] += 1
        result.append(matrix / counts[:, None])
    return result


class TapChannel(ComplexChannel):
    def __init__(self):
        super().__init__('identity')
        self.calls = 0
        self.last_symbols = None

    def forward(self, symbols, snr_db, generator=None):
        self.calls += 1
        self.last_symbols = symbols.detach().clone()
        return super().forward(symbols, snr_db, generator)


def fixture(h=64, w=128, requires_grad=False):
    rng = torch.Generator(device='cpu').manual_seed(1931)
    shapes = [(1, 32, h, w)] * 2 + [(1, 32, h//4, w//4)]
    return [torch.randn(shape, generator=rng).requires_grad_(requires_grad) for shape in shapes]


def original_link():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1932)
        model = StereoFeatureLink(channel='identity').eval()
    model.channel = TapChannel()
    return model


class Checks(unittest.TestCase):
    def test_dense_reference_random_both_directions_nondivisible_groups(self):
        rng = torch.Generator().manual_seed(1933)
        left = torch.randn((1, 32, 20, 68), generator=rng)
        right = torch.randn(left.shape, generator=rng)
        expected = dense_reference(left, right)
        for actual, reference in zip(c.correspondence(left, right), expected):
            np.testing.assert_allclose(actual.detach().numpy()[0], reference, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(actual.sum(-1), torch.ones((1, 32)), rtol=2e-6, atol=2e-6)
            vertical = torch.arange(32) // 8
            self.assertEqual(int(torch.count_nonzero(actual[0][vertical[:, None] != vertical[None, :]])), 0)

    def test_known_shift_borders_and_flat_features(self):
        left = torch.zeros((1, 32, 4, 16))
        for x in range(16):
            left[:, x, :, x] = 1
        right = torch.zeros_like(left); right[:, :, :, :-3] = left[:, :, :, 3:]
        left = left.repeat_interleave(4, 2).repeat_interleave(4, 3)
        right = right.repeat_interleave(4, 2).repeat_interleave(4, 3)
        lr, rl = c.correspondence(left, right)
        for actual, reference in zip((lr, rl), dense_reference(left, right)):
            np.testing.assert_allclose(actual.numpy()[0], reference, rtol=2e-5, atol=2e-6)
        self.assertGreater(float(lr[0, 5, 3:5].sum()), .99)
        self.assertGreater(float(rl[0, 1, 2:4].sum()), .99)
        for constant in (0., 1.):
            flat = torch.full_like(left, constant)
            lr, rl = c.correspondence(flat, flat)
            for actual, reference in zip((lr, rl), dense_reference(flat, flat)):
                np.testing.assert_allclose(actual.numpy()[0], reference, rtol=2e-5, atol=2e-6)
            self.assertEqual(float(lr[0, 0, 0]), 1.)
            self.assertEqual(float(rl[0, 7, 7]), 1.)

    def test_pixel_center_group_mean_no_overlapping_bins(self):
        feature = torch.arange(1 * 3 * 5 * 17, dtype=torch.float32).reshape(1, 3, 5, 17)
        output = c.group_mean(feature)
        membership = np.array([[int((y+.5)*4/5)*8 + int((x+.5)*8/17) for x in range(17)] for y in range(5)])
        for group in range(32):
            expected = feature.numpy()[0][:, membership == group].mean(1)
            np.testing.assert_array_equal(output.numpy()[0, group], expected)
        np.testing.assert_array_equal(c.group_ids(5, 17).numpy(), membership)
        self.assertEqual(int(np.bincount(membership.flatten()).sum()), 5*17)

    def test_shuffle_preserves_mass_entropy_and_vertical_group(self):
        left, right, _ = fixture()
        for matrix in c.correspondence(left, right):
            shuffled = c.shuffle_columns(matrix)
            self.assertTrue(torch.equal(torch.sort(matrix, -1).values, torch.sort(shuffled, -1).values))
            self.assertFalse(torch.equal(matrix, shuffled))
            self.assertTrue(torch.equal(c.shuffle_columns(shuffled), matrix))
            vertical = torch.arange(32) // 8
            self.assertEqual(int(torch.count_nonzero(shuffled[0][vertical[:, None] != vertical[None, :]])), 0)

    def test_initialization_no_global_rng_change_identical_arms(self):
        parent = original_link(); initial = torch.random.get_rng_state().clone()
        heads = []
        with patch.object(torch.cuda, 'manual_seed_all', side_effect=AssertionError('CUDA RNG must not be touched')):
            for arm in c.ARMS:
                model = c.install(copy.deepcopy(parent), arm)
                self.assertTrue(torch.equal(torch.random.get_rng_state(), initial))
                heads.append({k: v.clone() for k, v in model.gain_head.state_dict().items()})
                self.assertEqual(len(model.state_dict()), len(parent.state_dict()) + 4)
                self.assertEqual(sum(p.numel() for p in model.gain_head.parameters()), 561)
                self.assertTrue(all(p.requires_grad == (arm != 'U') for p in model.gain_head.parameters()))
                self.assertTrue(all(torch.equal(value, model.state_dict()[name]) for name, value in parent.state_dict().items()))
        for head in heads[1:]:
            self.assertTrue(all(torch.equal(v, head[k]) for k, v in heads[0].items()))
        self.assertEqual(int(torch.count_nonzero(heads[0]['2.weight'])), 0)
        self.assertEqual(int(torch.count_nonzero(heads[0]['2.bias'])), 0)

    def test_all_zero_heads_and_uniform_exact_parent_symbols_and_outputs(self):
        parent = original_link(); features = fixture(); batch = {'left_img': features[0][:, :3]}
        with torch.no_grad():
            expected = parent(*features, dict(batch))
            transmitted = parent.channel.last_symbols
            for arm in c.ARMS:
                model = c.install(copy.deepcopy(parent), arm)
                if arm == 'U':
                    for p in model.gain_head.parameters(): p.fill_(4.)
                before = model.channel.calls
                actual = model(*features, dict(batch))
                self.assertEqual(model.channel.calls - before, 1)
                self.assertTrue(torch.equal(model.channel.last_symbols, transmitted))
                self.assertTrue(all(torch.equal(a, e) for a, e in zip(actual, expected)))
                self.assertEqual(model.last_f9['amplitude_gains'], [1.] * 64)

    def test_dense_correspondence_has_both_input_gradients(self):
        rng = torch.Generator().manual_seed(1934)
        left = torch.randn((1, 32, 16, 64), generator=rng, requires_grad=True)
        right = torch.randn(left.shape, generator=rng, requires_grad=True)
        lr, rl = c.correspondence(left, right)
        weight = torch.randn(lr.shape, generator=rng)
        ((lr * weight).sum() + (rl * weight.flip(-1)).sum()).backward()
        for gradient in (left.grad, right.grad):
            self.assertIsNotNone(gradient); self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(torch.linalg.vector_norm(gradient)), 0.)

    def test_actual_codec_head_gradients_initial_zero_then_hidden_nonzero(self):
        rng = torch.Generator().manual_seed(1935)
        for arm in c.ARMS:
            model = c.install(original_link(), arm)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
            for step in range(2):
                features = fixture(requires_grad=True)
                outputs = model(*features, {'left_img': features[0][:, :3]})
                loss = sum((output * torch.randn(output.shape, generator=rng)).mean() for output in outputs)
                optimizer.zero_grad(set_to_none=True); loss.backward()
                for name, parameter in model.named_parameters():
                    if not parameter.requires_grad:
                        self.assertIsNone(parameter.grad); continue
                    self.assertIsNotNone(parameter.grad, (arm, step, name))
                    self.assertTrue(torch.isfinite(parameter.grad).all(), (arm, step, name))
                    if step == 0 and name.startswith('gain_head.0.'):
                        self.assertEqual(int(torch.count_nonzero(parameter.grad)), 0)
                    else:
                        self.assertGreater(float(torch.linalg.vector_norm(parameter.grad)), 0., (arm, step, name))
                for feature in features:
                    self.assertIsNotNone(feature.grad); self.assertTrue(torch.isfinite(feature.grad).all())
                    self.assertGreater(float(torch.linalg.vector_norm(feature.grad)), 0.)
                optimizer.step()

    def test_actual_native_layout_energy_and_one_channel_call(self):
        features = fixture(320, 1248)
        model = c.install(original_link(), 'P')
        # Decode's ordinary implementation is checked above; avoid allocating full native RGB-sized outputs here.
        seen = []
        def decode_only(self, received, public_shapes):
            seen.append((received.shape, public_shapes))
            return (received,)
        model.decode = types.MethodType(decode_only, model)
        with torch.no_grad():
            model.gain_head[2].weight.copy_(torch.linspace(-.3, .3, 16).reshape(1, 16))
            model.gain_head[2].bias.fill_(.25)
            result = model(*features, {'left_img': features[0][:, :3]})
        self.assertEqual(model.channel.calls, 1); self.assertEqual(len(seen), 1)
        self.assertEqual(tuple(result[0].shape), (1, 62400, 2))
        record = model.last_f9; account = model.last_accounting
        self.assertEqual(account['data_complex_uses'], 62400)
        self.assertEqual(account['stereo_complex_uses'], 49920)
        self.assertEqual(account['appearance_complex_uses'], 12480)
        self.assertEqual(account['total_complex_uses'], 62400)
        self.assertEqual(account['header_complex_uses'], 0); self.assertEqual(account['pilot_complex_uses'], 0)
        symbol = model.channel.last_symbols.numpy().astype(np.float64)
        power = np.square(symbol).sum(-1)[0]
        energies = []
        for view in range(2):
            grid = power[view*24960:(view+1)*24960].reshape(20, 1248)
            ids = np.array([[int((y+.5)*4/20)*8+int((x+.5)*8/1248) for x in range(1248)] for y in range(20)])
            energies.extend(float(grid[ids == g].sum()) for g in range(32))
        np.testing.assert_allclose(record['stereo_group_energy'], energies, rtol=1e-12, atol=1e-9)
        self.assertAlmostEqual(record['appearance_energy'], float(power[49920:].sum()), places=8)
        self.assertAlmostEqual(record['actual_energy_float64'], sum(energies)+record['appearance_energy'], places=8)
        self.assertLess(abs(record['actual_energy_float64']/62400-1), 1e-6)
        self.assertFalse(np.allclose(record['amplitude_gains'], 1))
        self.assertNotEqual(record['stereo_group_energy'], record['amplitude_gains'])
        self.assertFalse(record['receiver_side_information'])

    def test_inherited_receiver_uses_no_source_or_gain_parameters(self):
        model = c.install(original_link(), 'P'); features = fixture()
        shapes = tuple(tuple(x.shape) for x in features)
        with torch.no_grad():
            expected = model(*features, {'left_img': features[0][:, :3]})
            received = model.channel.last_symbols.clone()
        class Trap(nn.Module):
            def forward(self, *args, **kwargs): raise AssertionError('receiver used source-side module')
        model.stereo_encoder = Trap(); model.appearance_encoder = Trap(); model.gain_head = Trap()
        model.last_f9 = None
        with patch.object(c, 'correspondence', side_effect=AssertionError('receiver used sender matching')):
            with torch.no_grad(): actual = model.decode(received, shapes)
        self.assertIs(type(model).decode, StereoFeatureLink.decode)
        self.assertTrue(all(torch.equal(a, e) for a, e in zip(actual, expected)))
        with self.assertRaises(TypeError): model.decode(received, shapes, gains=torch.ones(64))

    def test_malformed_scopes_layouts_values_rejected(self):
        left, right, app = fixture()
        invalid = [(left[:, :31], right[:, :31]), (left.repeat(2, 1, 1, 1), right.repeat(2, 1, 1, 1)),
                   (left[:, :, :-1], right[:, :, :-1]), (left.double(), right.double()),
                   (left[:, :, :8], right[:, :, :8]), (left * float('nan'), right)]
        for a, b in invalid:
            with self.assertRaises(ValueError): c.correspondence(a, b)
        with self.assertRaises(ValueError): c.group_ids(3, 8)
        with self.assertRaises(ValueError): c.install(original_link(), 'bad')
        model = c.install(original_link(), 'G')
        with self.assertRaises(ValueError): c.install(model, 'P')
        with self.assertRaises(ValueError): model(left, right, app[:, :, :-1], {'left_img': left[:, :3]})
        model.snr_db = float('nan')
        with self.assertRaises(ValueError): model(left, right, app, {'left_img': left[:, :3]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('preserve previous CPU evidence: ' + str(args.output))
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    started = time.time()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    source_files = [Path(__file__).resolve(), Path(c.__file__).resolve(), ROOT/'experiments/geometry-link/F9-epipolar-conditioned-JSCC.md']
    report = dict(state='passed' if result.wasSuccessful() else 'failed', started_unix=started,
                  finished_unix=time.time(), tests_run=result.testsRun,
                  failures=[{'test': str(test), 'traceback': trace} for test, trace in result.failures],
                  errors=[{'test': str(test), 'traceback': trace} for test, trace in result.errors],
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                  torch_version=torch.__version__, numpy_version=np.__version__, torch_threads=torch.get_num_threads(),
                  scope='pure CPU coupling/core checks; no native detector, KITTI, GPU training or AP',
                  full_native_layout_sender_test=True, native_receiver_detector_executed=False)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('state', 'tests_run', 'scope')}))
    if not result.wasSuccessful(): raise SystemExit(1)


if __name__ == '__main__':
    main()
