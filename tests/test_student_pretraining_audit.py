import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('student_audit', ROOT/'scripts/audit_student_pretraining.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class StudentAuditTests(unittest.TestCase):
    def test_original_global_step_and_norm_buffers_have_no_exemption(self):
        sparse = torch.arange(48.).reshape(2, 2, 2, 2, 3)
        key = 'lidar_model.backbone_3d.conv.weight'
        reference = {key: sparse, 'global_step': torch.tensor(0),
                     'dense_head.norm_imitation.volume_features.scale': torch.tensor(1.)}
        actual = {**reference, key: sparse.permute(4, 0, 1, 2, 3).contiguous()}
        self.assertEqual(audit.frozen_original(reference, actual), [key])
        for changed in ('global_step', 'dense_head.norm_imitation.volume_features.scale'):
            with self.assertRaises(ValueError):
                audit.frozen_original(reference, {**actual, changed: actual[changed]+1})

    def test_saved_loss_and_frame_membership_are_independently_checked(self):
        interfaces = {name: dict(student_rms=0., teacher_rms=1., normalized_mse=1., cosine=0.)
                      for name in audit.INTERFACES}
        row = dict(step=1, epoch=1, frame_id='000003', loss=1., preclip_grad_norm=.1, interfaces=interfaces)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'training.jsonl'
            path.write_text(json.dumps(row)+'\n')
            self.assertEqual(audit.scalar_records(path, {'000003'}, 1, True)['steps'], 1)
            for bad in ({**row, 'frame_id': '000004'}, {**row, 'loss': .3}, {**row, 'step': 2}):
                path.write_text(json.dumps(bad)+'\n')
                with self.assertRaises(ValueError):
                    audit.scalar_records(path, {'000003'}, 1, True)


if __name__ == '__main__':
    unittest.main()
