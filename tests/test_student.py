import sys
from pathlib import Path
import unittest
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from geocomm.student import StereoStudentEncoder,teacher_feature_targets,feature_distillation


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3,32,1)
        self.bn = nn.BatchNorm2d(32)

    def forward(self,x):
        return [self.bn(self.conv(x))]


class Neck(nn.Module):
    def forward(self,features):
        return features[-1],torch.nn.functional.avg_pool2d(features[-1],4)


class StudentTests(unittest.TestCase):
    def test_frozen_teacher_state_and_training_gradient(self):
        torch.manual_seed(17)
        student,backbone,neck = StereoStudentEncoder(),Backbone(),Neck()
        backbone.train()
        # Preserve a deliberately mixed mode hierarchy.
        backbone.bn.eval()
        before = {name:value.clone() for name,value in backbone.state_dict().items()}
        modes = [module.training for module in backbone.modules()]
        left,right = torch.randn(1,3,16,32),torch.randn(1,3,16,32)
        ls,la = student(left)
        rs,_ = student(right)
        targets = teacher_feature_targets(backbone,neck,left,right)
        loss = feature_distillation((ls,rs,la),targets)
        loss.backward()
        self.assertEqual(modes,[module.training for module in backbone.modules()])
        for name,value in backbone.state_dict().items():
            torch.testing.assert_close(value,before[name],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in backbone.parameters()))
        self.assertTrue(all(not target.requires_grad for target in targets))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters()))
        student.eval()
        # Teacher data is not an inference argument; student retains full interfaces.
        with torch.no_grad():
            stereo,appearance = student(left)
        self.assertEqual(stereo.shape,(1,32,16,32))
        self.assertEqual(appearance.shape,(1,32,4,8))


if __name__=='__main__':
    unittest.main()
