import sys
from pathlib import Path
import unittest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from geocomm.feature_agreement import feature_agreement, assert_original_state


class FeatureAgreementTests(unittest.TestCase):
    def test_scale_collapse_and_direction_are_separated(self):
        target = torch.tensor([1., -1., 2., -2.])
        result = feature_agreement((2*target, torch.zeros_like(target), -target), (target,)*3)
        self.assertAlmostEqual(result['left_stereo']['normalized_mse'], 1.)
        self.assertAlmostEqual(result['left_stereo']['cosine'], 1.)
        self.assertAlmostEqual(result['left_stereo']['student_rms']/result['left_stereo']['teacher_rms'], 2.)
        self.assertEqual(result['right_stereo']['cosine'], 0.)
        self.assertEqual(result['right_stereo']['normalized_mse'], 1.)
        self.assertAlmostEqual(result['left_appearance']['cosine'], -1.)
        self.assertAlmostEqual(result['left_appearance']['normalized_mse'], 4.)
        with self.assertRaises(ValueError):
            feature_agreement((target,)*3, (torch.full_like(target, float('nan')),)*3)

    def test_frozen_check_detects_buffer_and_weight_mutation(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        reference = {k: v.clone() for k, v in model.state_dict().items()}
        self.assertEqual(assert_original_state(model, reference), len(reference))
        model[1].num_batches_tracked.add_(1)
        with self.assertRaisesRegex(RuntimeError, 'num_batches_tracked'):
            assert_original_state(model, reference)
        model.load_state_dict(reference)
        with torch.no_grad():
            model[0].weight.add_(1)
        with self.assertRaisesRegex(RuntimeError, '0.weight'):
            assert_original_state(model, reference)


if __name__ == '__main__':
    unittest.main()
