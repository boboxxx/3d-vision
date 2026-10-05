import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from geocomm.link import GeometryLink


class PosteriorTests(unittest.TestCase):
    def test_measured_half_bins_included_but_missing_depth_excluded(self):
        link=GeometryLink(cost_channels=2,appearance_channels=2)
        logits=torch.tensor([[[[2.,-1.,100.,100.]],[[0.,3.,-100.,-100.]]]],requires_grad=True)
        measured=torch.tensor([[[[2.01,3.59,0.,3.61]]]])
        samples=torch.tensor([2.4,3.2])
        loss=link._posterior_loss(logits,samples,measured)
        expected=torch.nn.functional.cross_entropy(torch.tensor([[2.,0.],[-1.,3.]]),torch.tensor([0,1]))
        torch.testing.assert_close(loss,expected)
        loss.backward()
        self.assertEqual(logits.grad[...,2:].abs().sum().item(),0)

    def test_raw_left_evidence_must_change_depth_distribution(self):
        # Raw concat repeats left evidence over depth; right evidence shifts.
        inputs = torch.tensor([[-2.,-2.,0.,1.],[2.,2.,0.,1.]]).reshape(2,2,2,1,1)
        linear = GeometryLink(cost_channels=2,appearance_channels=2)
        nonlinear = GeometryLink(cost_channels=2,appearance_channels=2,posterior_hidden=2)
        with torch.no_grad():
            linear.posterior.weight.fill_(1)
            linear.posterior.bias.zero_()
            first,last = nonlinear.posterior[0],nonlinear.posterior[2]
            first.weight.zero_(); first.bias.zero_(); last.weight.zero_(); last.bias.zero_()
            first.weight[0].fill_(1)
            last.weight[0,0].fill_(1)
        original = linear.posterior(inputs).squeeze(1).softmax(1)
        coupled = nonlinear.posterior(inputs).squeeze(1).softmax(1)
        torch.testing.assert_close(original[0],original[1],rtol=0,atol=1e-7)
        self.assertGreater((coupled[0]-coupled[1]).abs().max().item(),0.1)


if __name__=='__main__':
    unittest.main()
