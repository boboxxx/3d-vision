import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from geocomm.stereo_feature_link import StereoFeatureLink

class Tests(unittest.TestCase):
    def test_actual_gradient_energy_and_public_native_counts(self):
        torch.manual_seed(17);link=StereoFeatureLink(channel='identity').eval()
        features=[torch.randn(1,32,16,32),torch.randn(1,32,16,32),torch.randn(1,32,4,8)];batch={'left_img':torch.zeros(1,3,16,32)}
        received=link(*features,batch)
        loss=.5*sum((r-t).square().mean() for r,t in zip(received[:2],features[:2]))+(received[2]-features[2]).square().mean()
        loss.backward();self.assertEqual(len(list(link.parameters())),16)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in link.parameters()))
        self.assertTrue(all(r.shape==t.shape for r,t in zip(received,features)))
        account=link.last_accounting;self.assertEqual(account['total_complex_uses'],80)
        self.assertAlmostEqual(account['tx_energy_per_frame'].item(),80,places=3)
        layout=link.layout(((1,32,320,1248),(1,32,320,1248),(1,32,80,312)));self.assertEqual(sum(c*h*w//2 for _,c,h,w in layout),62400)
        self.assertEqual(layout,[(1,2,20,1248),(1,2,20,1248),(1,8,20,156)])
        self.assertAlmostEqual(account['cbr_complex_per_input_real_scalar'],1/38.4)
    def test_receiver_replay_and_no_clean_residual(self):
        torch.manual_seed(19);link=StereoFeatureLink(channel='identity').eval();features=[torch.randn(1,32,16,32),torch.randn(1,32,16,32),torch.randn(1,32,4,8)]
        payload=[]
        hook=link.channel.register_forward_pre_hook(lambda m,inputs:payload.append(inputs[0].detach().clone()))
        output=link(*features,{'left_img':torch.zeros(1,3,16,32),'depth_gt_img':torch.full((1,),float('nan'))});hook.remove()
        replay=link.decode(payload[0],tuple(tuple(x.shape) for x in features));self.assertTrue(all(torch.equal(a,b) for a,b in zip(output,replay)))
        for decoder in [link.stereo_decoder,link.appearance_decoder]:
            for p in decoder.parameters(): p.data.zero_()
        zero=link(*features,{'left_img':torch.zeros(1,3,16,32)})
        self.assertTrue(all(not r.any() for r in zero))
        with self.assertRaisesRegex(ValueError,'duration'): link.decode(payload[0][:,:-1],tuple(tuple(x.shape) for x in features))
        with self.assertRaises(TypeError): link.decode(payload[0],tuple(tuple(x.shape) for x in features),features)
    def test_untransmitted_teacher_data_and_malformed_shapes(self):
        torch.manual_seed(23);link=StereoFeatureLink(channel='identity').eval();features=[torch.randn(1,32,16,32),torch.randn(1,32,16,32),torch.randn(1,32,4,8)]
        first=link(*features,{'left_img':torch.zeros(1,3,16,32)})
        second=link(*features,{'left_img':torch.zeros(1,3,16,32),'gt_boxes':torch.ones(99),'points':torch.ones(5)})
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(first,second)))
        with self.assertRaises(ValueError): link(features[0],features[1][:,:,:,:-1],features[2],{'left_img':torch.zeros(1,3,16,32)})
        with self.assertRaisesRegex(ValueError,'quarter-resolution'):
            link(features[0],features[1],features[0],{'left_img':torch.zeros(1,3,16,32)})
        features[0].fill_(float('nan'))
        with self.assertRaisesRegex(ValueError,'energy'): link(*features,{'left_img':torch.zeros(1,3,16,32)})

if __name__=='__main__': unittest.main()
