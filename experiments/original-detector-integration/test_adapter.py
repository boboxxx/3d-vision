"""Engineering interface/physical accounting tests, no trained-model AP claim."""
import importlib.util
from pathlib import Path
import sys
import unittest
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'reproduction/cao2025'))
import radio
from wireless import WirelessVariant
from adapter import transmit_rgb,stereo_rcnn_preprocess
spec=importlib.util.spec_from_file_location('official_stereo_blob',ROOT/'third_party/Stereo-RCNN/lib/model/utils/blob.py')
blob=importlib.util.module_from_spec(spec);spec.loader.exec_module(blob)

class SemanticFixture(nn.Module):
 def encode(self,left,right,masks):
  shape=tuple(left.shape[-2:]);h,w=shape
  images=[F.pad(x,(0,(-w)%6,0,(-h)%6),mode='replicate') for x in (left,right)]
  padded=[F.pad(m,(0,(-w)%6,0,(-h)%6)) for m in masks]
  return dict(global_values=tuple(F.avg_pool2d(x,6) for x in images),
   key_values=tuple(F.avg_pool2d(x*m,2) for x,m in zip(images,padded)),masks=tuple(padded),original_shape=shape)
 def decode(self,payload):
  return tuple(F.interpolate(x,size=payload['original_shape'],mode='bilinear',align_corners=False) for x in payload['global_values'])

class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(2)
 def test_actual_original_wire_received_only_no_reconstruction_target_and_states_fixed(self):
  torch.manual_seed(17);model=WirelessVariant(SemanticFixture()).eval();left=torch.rand(1,3,198,204);right=torch.rand_like(left)
  boxes=[[dict(xyxy=[10,20,38,60],confidence=.8)],[]]
  before={name:value.clone() for name,value in model.state_dict().items()}
  # Broad original forward/semantic-MSE are forbidden for this adapter.
  model.forward=lambda *a,**k:(_ for _ in ()).throw(AssertionError('broad model.forward'))
  model.semantic_mse=lambda *a,**k:(_ for _ in ()).throw(AssertionError('clean reconstruction target read'))
  receive=model.receive_payload;observations=[]
  def checked_receive(observation):
   observations.append(observation);return receive(observation)
  model.receive_payload=checked_receive
  with torch.no_grad():r=transmit_rgb(model,radio,left,right,boxes,'identity')
  self.assertIsNone(r['erasure']);self.assertEqual(len(observations),1);self.assertIsInstance(observations[0],radio.Observation)
  self.assertTrue(all(x.shape==left.shape for x in r['outputs']));self.assertGreater(r['accounting']['control_uses'],0)
  self.assertEqual(r['accounting']['pilot_uses'],0);self.assertEqual(r['accounting']['key_cells'][1],0)
  self.assertTrue(all(torch.equal(before[k],v) for k,v in model.state_dict().items()))
 def test_control_CRC_erasure_retains_full_attempt_resources_and_never_decodes(self):
  model=WirelessVariant(SemanticFixture()).eval();image=torch.rand(1,3,198,204)
  def corrupt(air,channel,snr_db,generator):
   observation=radio.propagate(air,channel,snr_db,generator);symbols=observation.symbols.clone()
   symbols[:7,0]*=-1;return replace(observation,symbols=symbols)
  fake=SimpleNamespace(roi_masks=radio.roi_masks,pilot_count=radio.pilot_count,propagate=corrupt,FrameErasure=radio.FrameErasure)
  model.semantic.decode=lambda payload:(_ for _ in ()).throw(AssertionError('decode on erasure'))
  with torch.no_grad():r=transmit_rgb(model,fake,image,image,[[],[]],'identity')
  self.assertIsNone(r['outputs']);self.assertIsNone(r['decoder_range']);self.assertIn('CRC',r['erasure'])
  self.assertGreater(r['accounting']['total_uses'],r['accounting']['control_uses']);self.assertGreater(r['accounting']['total_energy'],0)
 def test_clean_relay_matches_official_BGR_preprocessing_including_native_crop(self):
  rng=np.random.default_rng(17);rgb=rng.integers(0,256,(198,224,3),dtype=np.uint8)
  decoded=torch.from_numpy(rgb.copy()).permute(2,0,1).unsqueeze(0).float()/255.
  means=np.array([[[102.9801,115.9465,122.7717]]],dtype=np.float32)
  expected=blob.prep_im_for_blob(rgb[...,::-1].copy(),rgb[...,::-1].copy(),means,224,240)
  images,info,shape=stereo_rcnn_preprocess((decoded,decoded),blob.prep_im_for_blob,means,224,240)
  self.assertEqual(shape,rgb.shape);self.assertEqual(images[0].shape[-1],240)
  for actual,want in zip(images,expected[:2]):np.testing.assert_array_equal(actual.numpy()[0].transpose(1,2,0),want)
  self.assertAlmostEqual(float(info[0,2]),expected[2],places=6)
 def test_decoded_float_range_is_preserved_and_invalid_input_stops(self):
  image=torch.empty(1,3,198,204);image[:,0]=-.25;image[:,1]=1.25;image[:,2]=.5
  images,_,_=stereo_rcnn_preprocess((image,image),blob.prep_im_for_blob,np.zeros((1,1,3),np.float32),198,300)
  self.assertAlmostEqual(float(images[0][0,0,0,0]),127.5);self.assertAlmostEqual(float(images[0][0,1,0,0]),318.75)
  self.assertAlmostEqual(float(images[0][0,2,0,0]),-63.75)
  model=WirelessVariant(SemanticFixture()).eval()
  with self.assertRaisesRegex(RuntimeError,'no_grad'):transmit_rgb(model,radio,image,image,[[],[]],'identity')
  with torch.no_grad():
   with self.assertRaises(ValueError):transmit_rgb(model,radio,image,image,[[],[]],'identity')

if __name__=='__main__':unittest.main()
