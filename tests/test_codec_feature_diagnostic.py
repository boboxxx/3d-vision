import importlib.util
from pathlib import Path
import unittest
import torch

spec = importlib.util.spec_from_file_location('codec_diagnostic',
    Path(__file__).resolve().parents[1]/'scripts/diagnose_codec_features.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FeatureDistortionTests(unittest.TestCase):
    def test_exact_sign_and_scale_with_independent_analytic_values(self):
        a = torch.tensor([-2., -1., 0., 1., 2.])
        exact = module.compare_features(a, a, chunk_size=2)
        self.assertEqual(exact['nmse'], 0.)
        self.assertEqual(exact['cosine'], 1.)
        altered = module.compare_features(a, -2*a, chunk_size=3)
        self.assertEqual(altered['mse'], 18.)
        self.assertEqual(altered['nmse'], 9.)
        self.assertEqual(altered['cosine'], -1.)
        self.assertEqual(altered['reference_negative_fraction'], .4)
        self.assertEqual(altered['received_rms'], 2*exact['reference_rms'])

    def test_undefined_zero_norms_and_invalid_values(self):
        result = module.compare_features(torch.zeros(3), torch.ones(3), chunk_size=1)
        self.assertIsNone(result['nmse'])
        self.assertIsNone(result['cosine'])
        self.assertEqual(result['mse'], 1.)
        for a, b in [(torch.zeros(2), torch.zeros(3)),
                     (torch.ones(2), torch.tensor([1., float('nan')])),
                     (torch.zeros(0), torch.zeros(0))]:
            with self.assertRaises(ValueError):
                module.compare_features(a, b)


if __name__ == '__main__':
    unittest.main()
