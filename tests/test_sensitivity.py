import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from geocomm.sensitivity import TaskSensitivity
from geocomm.link import GeometryLink


class SensitivityTests(unittest.TestCase):
    def test_native_detection_gradient_excludes_imitation_and_second_order(self):
        # Execute the patched actual upstream get_loss in isolation, with known
        # native losses and a deliberately overwhelming imitation auxiliary.
        checkout = ROOT/'third_party/LIGA-Stereo'
        if not checkout.exists():
            self.skipTest('official checkout unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for path in ['liga/models/dense_heads/anchor_head_template.py','liga/models/detectors_stereo/liga.py']:
                original = subprocess.run(['git','-C',str(checkout),'show','HEAD:'+path],
                    check=True,capture_output=True).stdout
                target = root/path; target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(original)
            # The detector anchor only needs the preceding optional link loss.
            detector = root/'liga/models/detectors_stereo/liga.py'
            detector.write_text(detector.read_text().replace('        loss = loss_rpn + loss_depth\n',
                '        loss = loss_rpn + loss_depth\n        # GEOCOMM_TASK_LOSS\n'))
            spec = importlib.util.spec_from_file_location('sensitivity_patch',ROOT/'scripts/patch_sensitivity_liga.py')
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); module.patch(root)
            text = (root/'liga/models/dense_heads/anchor_head_template.py').read_text()
            method = text.split('    def get_loss(self):',1)[1].split('    def generate_predicted_boxes',1)[0]
            namespace = {}; exec('class Head:\n    def get_loss(self):'+method,namespace)
        received = torch.arange(1.,9.).reshape(1,4,2).requires_grad_()
        weights = torch.tensor([1.,2.,3.,4.]).reshape(1,4,1).expand_as(received)
        native = (received*weights).sum(); imitation = 1000*received.square().sum()
        head = namespace['Head'](); head.communication_sensitivity_enabled = True
        head.forward_ret_dict = dict(imitation_features_pairs=[dict(pred=None,gt=None,
            config=None,stereo_feature_name='fake')])
        head.get_cls_layer_loss = lambda:(native*.4,{})
        head.get_box_reg_layer_loss = lambda:(native*.6,{})
        head.get_imitation_reg_layer_loss = lambda **kw:(imitation,{})
        total, _ = head.get_loss()
        predictor = TaskSensitivity(4,4)
        cost = torch.randn(1,4,1,1,2,requires_grad=True)
        appearance = torch.randn(1,4,1,2,requires_grad=True)
        logits = predictor(cost,appearance)
        auxiliary,target_log = predictor.distillation_loss(
            head.forward_ret_dict['communication_detection_loss'],received,logits)
        expected = weights.square().sum(-1)
        torch.testing.assert_close(target_log,(expected/expected.mean()).log())
        (total+auxiliary).backward()
        torch.testing.assert_close(received.grad,weights+2000*received.detach())
        self.assertIsNone(cost.grad); self.assertIsNone(appearance.grad)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in predictor.parameters()))

    def test_inference_uses_no_gradient_teacher_or_groundtruth(self):
        torch.manual_seed(17)
        link = GeometryLink(4,4,2,task_sensitivity=True,allocation='geometry_task').eval()
        cost,appearance = torch.randn(1,4,6,8,10),torch.randn(1,4,16,20)
        depth = torch.linspace(5,65,6)
        first_batch = dict(left_img=torch.zeros(1,3,32,40),depth_gt_img=torch.randn(1,1,32,40))
        second_batch = dict(left_img=first_batch['left_img'])
        with torch.no_grad():
            torch.manual_seed(43); first = link(cost,appearance,first_batch,depth)
            torch.manual_seed(43); second = link(cost,appearance,second_batch,depth)
        for a,b in zip(first,second):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertNotIn('communication_sensitivity_context',first_batch)
        self.assertEqual(link.last_accounting['total_complex_uses'],52)
        self.assertAlmostEqual(link.last_accounting['tx_energy_per_frame'].item(),52,places=3)


if __name__=='__main__':
    unittest.main()
