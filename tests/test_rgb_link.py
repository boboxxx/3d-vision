import unittest
from pathlib import Path
import sys
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from geocomm.rgb_link import RGBLink


class RGBTests(unittest.TestCase):
    def test_payload_energy_pilots_and_no_clean_bypass(self):
        torch.manual_seed(17)
        link = RGBLink(complex_width=10,channel='rayleigh',pilots=8).eval()
        left,right = torch.randn(1,3,37,63),torch.randn(1,3,37,63)
        metadata = {}
        outputs = link(left,right,metadata)
        self.assertEqual([tuple(t.shape) for t in outputs],[tuple(left.shape),tuple(right.shape)])
        accounting = metadata['communication_accounting']
        self.assertEqual(accounting['data_complex_uses'],120)
        self.assertEqual(accounting['total_complex_uses'],128)
        torch.testing.assert_close(accounting['tx_energy_per_frame'],torch.tensor([128.]))
        original = link.channel
        class NoReceivedSignal(nn.Module):
            def forward(self,symbols,snr):
                _,account = original(symbols,snr)
                return torch.zeros_like(symbols),account
        link.channel = NoReceivedSignal()
        first = link(left,right,{})
        second = link(torch.randn_like(left)*5,torch.randn_like(right)*5,{})
        for a,b in zip(first,second):
            torch.testing.assert_close(a,b,rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
