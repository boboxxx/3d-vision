"""Exact native AST crop/collate bodies; no CUDA/model or AP substitution."""
import ast,copy,importlib.util,sys,unittest
from collections import defaultdict
from functools import partial
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from adapter import liga_preprocess
from types import SimpleNamespace

class Config(dict):
 def __getattr__(self,name):return self[name]

def native_bodies():
 namespace=dict(np=np,defaultdict=defaultdict)
 for path,class_name,names in [
  ('liga/datasets/augmentor/stereo_data_augmentor.py','StereoDataAugmentor',{'pre_2d_transformation','random_crop','forward'}),
  ('liga/datasets/stereo_dataset_template.py','StereoDatasetTemplate',{'collate_batch'})]:
  source=ast.parse((ROOT/'third_party/LIGA-Stereo'/path).read_text())
  cls=next(node for node in source.body if isinstance(node,ast.ClassDef) and node.name==class_name)
  cls=copy.deepcopy(cls);cls.bases=[];cls.body=[node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name in names]
  if len(cls.body)!=len(names):raise AssertionError('native definitions missing')
  exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),path,'exec'),namespace)
 aug=namespace['StereoDataAugmentor']();cfg=Config(MIN_REL_X=0,MAX_REL_X=0,MIN_REL_Y=1.,MAX_REL_Y=1.,MAX_CROP_H=320,MAX_CROP_W=1280)
 aug.data_augmentor_queue=[partial(aug.random_crop,config=cfg)]
 spec=importlib.util.spec_from_file_location('native_calibration',ROOT/'third_party/LIGA-Stereo/liga/utils/calibration_kitti.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 K=np.array([[700.,0,620.],[0,700.,185.],[0,0,1.]],dtype=np.float32)
 P2=np.column_stack((K,np.zeros(3,np.float32)));P3=P2.copy();P3[0,3]=-378.
 calib=module.Calibration(dict(P2=P2,P3=P3,R0=np.eye(3,dtype=np.float32),Tr_velo2cam=np.zeros((3,4),np.float32)))
 return aug,namespace['StereoDatasetTemplate'].collate_batch,calib

class Tests(unittest.TestCase):
 def test_native_crop_calibration_original_shape_and_padding(self):
  aug,collate,calib=native_bodies();original=calib.P2.copy()
  rgb=torch.zeros(1,3,375,1242);rgb[:,:,55,0]=1.
  batch=liga_preprocess((rgb,rgb),calib,'000036',aug,collate)
  self.assertEqual(tuple(batch['left_img'].shape),(1,3,320,1248));self.assertEqual(batch['image_shape'].tolist(),[[375,1242]])
  self.assertEqual(batch['calib'][0].offsets,[0,55]);self.assertEqual(batch['calib_ori'][0].offsets,[0,0])
  np.testing.assert_array_equal(calib.P2,original)
  self.assertAlmostEqual(batch['calib'][0].cv,130.,places=5)
  self.assertTrue(torch.equal(batch['left_img'][...,-6:],torch.zeros(1,3,320,6)))
  self.assertAlmostEqual(float(batch['left_img'][0,0,0,0]),(1.-.485)/.229,places=6)
  again=liga_preprocess((rgb,rgb),calib,'000036',aug,collate)
  np.testing.assert_array_equal(again['calib'][0].P2,batch['calib'][0].P2)
 def test_center_horizontal_crop_uses_native_intrinsics_offset(self):
  aug,collate,calib=native_bodies();rgb=torch.zeros(1,3,330,1300)
  batch=liga_preprocess((rgb,rgb),calib,'fixture',aug,collate)
  self.assertEqual(batch['calib'][0].offsets,[10,10]);self.assertEqual(batch['image_shape'].tolist(),[[330,1300]])
  self.assertEqual(tuple(batch['left_img'].shape),(1,3,320,1280))
  self.assertAlmostEqual(batch['calib'][0].cu,610.,places=5)
 def test_float_decoded_range_preserved_and_bad_geometry_rejected(self):
  aug,collate,calib=native_bodies();rgb=torch.full((1,3,375,1242),1.25)
  batch=liga_preprocess((rgb,rgb),calib,'fixture',aug,collate)
  self.assertAlmostEqual(float(batch['left_img'][0,0,0,0]),(1.25-.485)/.229,places=6)
  aug.data_augmentor_queue[0].keywords['config']['MAX_CROP_H']=300
  with self.assertRaises(ValueError):liga_preprocess((rgb,rgb),calib,'fixture',aug,collate)

if __name__=='__main__':unittest.main()
