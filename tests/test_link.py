import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from geocomm.channel import ComplexChannel
from geocomm.allocation import allocate_power, depth_moments
from geocomm.link import GeometryLink


class PhysicsTests(unittest.TestCase):
    def test_awgn_variance_and_accounting(self):
        torch.manual_seed(17)
        x = torch.ones(1, 200000, 2) / (2 ** .5)
        y, a = ComplexChannel()(x, 10.)
        self.assertAlmostEqual((y-x).square().mean().item(), .05, delta=.0007)
        self.assertEqual(a["total_complex_uses"], 200000)
        self.assertAlmostEqual(a["tx_energy_per_frame"].item(), 200000, delta=1)

    def test_fading_uses_pilots_and_has_no_oracle_argument(self):
        x = torch.ones(2, 4096, 2) / (2 ** .5)
        ch = ComplexChannel("rayleigh", pilots=8)
        y, a = ch(x, 20.)
        self.assertTrue(torch.isfinite(y).all())
        self.assertEqual(a["total_complex_uses"], 4104)
        self.assertEqual(a["receiver_csi"], "noisy_pilots")
        self.assertTrue(torch.allclose(a["tx_energy_per_frame"], torch.full((2,), 4104.)))
        with self.assertRaises(ValueError):
            ComplexChannel("rayleigh", 0)

    def test_kkt_and_budget(self):
        a = torch.tensor([[.01, 1., 5., 20.], [1., 1., 1., 1.]])
        power = allocate_power(a, 2.)
        self.assertTrue(torch.allclose(power.sum(1), torch.tensor([4., 4.]), atol=1e-5))
        self.assertTrue(torch.all(power >= 0))
        self.assertTrue(torch.allclose(power[1], torch.ones(4), atol=1e-6))
        derivative = a[0]*2/(1+2*power[0]).square()
        active = power[0] > 1e-6
        self.assertLess((derivative[active].max()-derivative[active].min()).item(), 1e-5)
        self.assertTrue((derivative[~active] <= derivative[active].mean()+1e-6).all())
        self.assertLess((a[0]/(1+2*power[0])).sum().item(), (a[0]/3).sum().item())

    def test_depth_moments_multimodal(self):
        p = torch.tensor([.5, .5]).reshape(1, 2, 1, 1)
        mean, variance = depth_moments(p, torch.tensor([10., 70.]))
        self.assertEqual(mean.item(), 40.)
        self.assertEqual(variance.item(), 900.)


class LinkTests(unittest.TestCase):
    def inputs(self):
        torch.manual_seed(29)
        cost = torch.randn(1, 4, 6, 8, 10, requires_grad=True)
        app = torch.randn(1, 4, 16, 20, requires_grad=True)
        batch = {"left_img": torch.zeros(1, 3, 32, 40),
                 "depth_gt_img": torch.zeros(1, 1, 32, 40)}
        batch["depth_gt_img"][0, 0, 10, 12] = 21.
        return cost, app, batch, torch.linspace(5, 65, 6)

    def test_task_gradient_and_no_eval_groundtruth(self):
        cost, app, batch, z = self.inputs()
        link = GeometryLink(4, 4, 2)
        c, a = link(cost, app, batch, z)
        loss = c.square().mean() + a.square().mean() + batch["communication_aux_loss"]
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(link.cost_encoder[1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(link.posterior.weight.grad.abs().sum().item(), 0)
        self.assertEqual(c.shape, cost.shape)
        self.assertEqual(a.shape, app.shape)
        # ceil(6/8)*ceil(8/4)*ceil(10/4) + ceil(16/4)*ceil(20/4), width 2
        account = link.last_accounting
        self.assertEqual(account["total_complex_uses"], (1*2*3+4*5)*2)
        self.assertAlmostEqual(account["tx_energy_per_frame"].item(), 52., places=3)
        link.eval()
        torch.manual_seed(43)
        first = link(cost.detach(), app.detach(), batch, z)
        del batch["depth_gt_img"]
        batch.pop("communication_aux_loss")
        torch.manual_seed(43)
        second = link(cost.detach(), app.detach(), batch, z)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))

    def test_no_clean_residual_bypass(self):
        cost, app, batch, z = self.inputs()
        link = GeometryLink(4, 4, 2, channel="identity", allocation="uniform").eval()
        for module in [link.cost_encoder, link.cost_decoder,
                       link.appearance_encoder, link.appearance_decoder]:
            for p in module.parameters():
                p.data.zero_()
        c, a = link(cost, app, batch, z)
        self.assertEqual(c.abs().sum().item(), 0)
        self.assertEqual(a.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
