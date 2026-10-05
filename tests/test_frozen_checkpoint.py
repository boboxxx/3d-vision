import importlib.util
from pathlib import Path
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('checkpoint_audit',ROOT/'scripts/audit_codec_checkpoint.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FrozenCheckpointTests(unittest.TestCase):
    def test_catches_frozen_weight_bn_and_step_changes(self):
        reference = {'receiver.weight': torch.tensor([1.,2.]),
                     'backbone_3d.student_semantic_link_encoder.weight': torch.tensor([.7]),
                     'receiver.running_mean': torch.tensor([.3,.4]),
                     'global_step': torch.tensor([10]),
                     'dense_head.norm_imitation.volume_features.scale': torch.tensor([[2.]])}
        actual = {k:v.clone() for k,v in reference.items()}
        actual['global_step'] += 3
        actual['dense_head.norm_imitation.volume_features.scale'] *= .9
        self.assertEqual(module.frozen_state(reference,actual,3)['unchanged_original_tensors'],3)
        for key in ['receiver.weight','receiver.running_mean','global_step',
                    'backbone_3d.student_semantic_link_encoder.weight']:
            corrupt = {k:v.clone() for k,v in actual.items()}
            corrupt[key] += 1
            with self.assertRaises(ValueError):
                module.frozen_state(reference,corrupt,3)

    def test_sparse_values_cannot_be_dropped_or_changed(self):
        key = 'lidar_model.backbone_3d.conv.weight'
        original = torch.arange(2*3*4*5*6).reshape(2,3,4,5,6).float()
        actual = original.permute(4,0,1,2,3).contiguous()
        report = module.frozen_state({key:original},{key:actual},1)
        self.assertEqual(report['value_preserving_sparse_layout_keys'],[key])
        actual.flatten()[0] += 1
        with self.assertRaises(ValueError):
            module.frozen_state({key:original},{key:actual},1)


if __name__ == '__main__':
    unittest.main()
