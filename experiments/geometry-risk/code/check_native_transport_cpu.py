"""Analytic receiver gradient, identity, actual perturbation and isolation fixtures."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from native_transport import RiskChannel,array_identity,global_rng_identity


class IdentityChannel(torch.nn.Module):
    kind='identity'
    def forward(self,symbols,snr_db,generator=None):
        return symbols,dict(data_complex_uses=symbols.shape[1],total_complex_uses=symbols.shape[1],tx_energy_per_frame=symbols.square().sum((1,2)).detach())


class TransportChecks(unittest.TestCase):
    def test_actual_symbol_leaf_gradient_matches_analytic_linear_receiver_with_no_parameter_gradient(self):
        link=torch.nn.Module();link.channel=IdentityChannel();link.eval()
        parameter=torch.nn.Parameter(torch.tensor(2.),requires_grad=False)
        original=(torch.arange(128,dtype=torch.float32).reshape(1,64,2)/50)*parameter
        weights=torch.linspace(-2,3,128).reshape(1,64,2)
        with tempfile.TemporaryDirectory() as d:
            transport=RiskChannel(link,d,[(1,2,4,16)])
            transport.prepare('000000','clean');output,_=link.channel(original,10.)
            self.assertTrue(torch.equal(output,original));self.assertFalse(original.requires_grad)
            loss=(output*weights).sum();gradient,=torch.autograd.grad(loss,transport.leaf)
            torch.testing.assert_close(gradient,weights,atol=0,rtol=0)
            self.assertIsNone(parameter.grad);record=transport.finish()
            self.assertEqual(record['PCG64']['before'],record['PCG64']['after'])
            self.assertEqual(record['transmitted_symbols_sha256'],record['received_symbols_sha256'])
            transport.close()

    def test_only_selected_cell_changes_full_duration_and_exact_noise_replay_with_no_global_draw(self):
        link=torch.nn.Module();link.channel=IdentityChannel();link.eval()
        original=torch.linspace(-1,1,128).reshape(1,64,2)
        before=global_rng_identity()
        with tempfile.TemporaryDirectory() as d:
            transport=RiskChannel(link,d,[(1,2,4,16)])
            transport.prepare('000000','clean');link.channel(original,10.);transport.finish()
            transport.prepare('000000','group',5,0);received,account=link.channel(original,10.);record=transport.finish()
            noise=np.load(record['noise_path'],allow_pickle=False)
            state=np.random.PCG64();state.state=record['PCG64']['before'];rng=np.random.Generator(state)
            expected=(rng.standard_normal((64,2))*np.sqrt(.1/2));expected[transport.group_ids!=5]=0;expected=expected.astype(np.float32)
            np.testing.assert_array_equal(noise,expected);self.assertEqual(rng.bit_generator.state,record['PCG64']['after'])
            mask=torch.as_tensor(transport.group_ids!=5)
            self.assertTrue(torch.equal(received[:,mask],original[:,mask]))
            torch.testing.assert_close(received,original+torch.from_numpy(noise)[None],atol=0,rtol=0)
            self.assertEqual(account['total_complex_uses'],64)
            self.assertEqual(record['global_rng_before'],record['global_rng_after'])
            self.assertEqual(global_rng_identity(),before);transport.close()

    def test_unprepared_overlapping_and_wrong_symbol_calls_refused(self):
        link=torch.nn.Module();link.channel=IdentityChannel();link.eval()
        with tempfile.TemporaryDirectory() as d:
            t=RiskChannel(link,d,[(1,2,4,16)])
            with self.assertRaises(RuntimeError):link.channel(torch.zeros(1,64,2),10.)
            t.prepare('000000','clean')
            with self.assertRaises(ValueError):t.prepare('000000','clean')
            with self.assertRaises(ValueError):link.channel(torch.zeros(1,63,2),10.)
            link.channel(torch.ones(1,64,2),10.)
            with self.assertRaises(RuntimeError):link.channel(torch.ones(1,64,2),10.)
            t.finish();t.close()


if __name__=='__main__':unittest.main(verbosity=2)
