"""Exact pure author writer definitions, no detector/GPU model construction."""
import ast,hashlib,types
from pathlib import Path
import numpy as np
import torch

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(root):
 calibration=root/'third_party/LIGA-Stereo/liga/utils/calibration_kitti.py'
 g=dict(np=np,torch=torch);exec(compile(calibration.read_text(),str(calibration),'exec'),g)
 Calibration=g['Calibration']
 boxpath=root/'third_party/LIGA-Stereo/liga/utils/box_utils.py'
 wanted={'boxes3d_lidar_to_kitti_camera','boxes3d_to_grid3d_kitti_camera','boxes3d_kitti_camera_to_imageboxes'}
 functions=[node for node in ast.parse(boxpath.read_text()).body if isinstance(node,ast.FunctionDef) and node.name in wanted]
 assert len(functions)==len(wanted)
 bg=dict(np=np,Calibration=Calibration);exec(compile(ast.Module(body=functions,type_ignores=[]),str(boxpath),'exec'),bg)
 writerpath=root/'third_party/LIGA-Stereo/liga/datasets/kitti/stereo_kitti_dataset.py'
 classes=[node for node in ast.parse(writerpath.read_text()).body if isinstance(node,ast.ClassDef) and node.name=='StereoKittiDataset'];assert len(classes)==1
 functions=[node for node in classes[0].body if isinstance(node,ast.FunctionDef) and node.name=='generate_prediction_dicts'];assert len(functions)==1
 wg=dict(np=np,torch=torch,box_utils=types.SimpleNamespace(**{k:bg[k] for k in wanted}))
 exec(compile(ast.Module(body=functions,type_ignores=[]),str(writerpath),'exec'),wg)
 return Calibration,wg['generate_prediction_dicts'],{str(p.relative_to(root)):sha(p) for p in (calibration,boxpath,writerpath)}

def replay(root,arrays,frame,directory):
 Calibration,writer,_=load(root)
 public=dict(P2=arrays['original_P2'],P3=arrays['original_P3'],R0=np.eye(3,dtype=np.float32),Tr_velo2cam=np.zeros((3,4),np.float32))
 original=Calibration(public)
 cropped=Calibration(dict(public,P2=arrays['cropped_P2'],P3=arrays['cropped_P3']))
 cropped.offsets=arrays['crop_offsets'].tolist()
 batch=dict(frame_id=[frame],calib=[cropped],calib_ori=[original],image_shape=arrays['image_shape'])
 predictions=[{k:torch.from_numpy(arrays[k].copy()) for k in ('pred_boxes','pred_scores','pred_labels')}]
 writer(types.SimpleNamespace(boxes_gt_in_cam2_view=False),batch,predictions,['Car','Pedestrian','Cyclist'],output_path=directory)
 return (directory/(frame+'.txt')).read_bytes()
