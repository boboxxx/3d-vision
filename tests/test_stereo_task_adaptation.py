import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
import torch
from torch import nn
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from geocomm.stereo_feature_link import StereoFeatureLink
from geocomm.stereo_task_adaptation import NativeTaskAdaptation,freeze_except_codec,assert_frozen,SEQUENCE
spec=importlib.util.spec_from_file_location('F6_audit',ROOT/'scripts/audit_stereo_native_task.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

class Student(nn.Module):
 def __init__(self):super().__init__();self.conv=nn.Conv2d(3,32,1)
 def forward(self,x):
  stereo=self.conv(x);return stereo,torch.nn.functional.avg_pool2d(stereo,4)+1
class Cost(nn.Module):
 def __init__(self):super().__init__();self.detach=False
 def forward(self,left,right,*args):
  output=torch.cat((left,right),dim=1).unsqueeze(2)
  return output.detach() if self.detach else output
class Backbone(nn.Module):
 def __init__(self):
  super().__init__();self.semantic_link=None;self.stereo_feature_link=StereoFeatureLink(channel='identity')
  self.student_semantic_link_encoder=Student();self.build_cost=Cost();self.feature_backbone=nn.Identity();self.feature_neck=nn.Identity()
 def forward(self,batch):
  l,a=self.student_semantic_link_encoder(batch['left_img']);r,_=self.student_semantic_link_encoder(batch['right_img'])
  l,r,a=self.stereo_feature_link(l,r,a,batch);c=self.build_cost(l,r,None,None,None)
  batch['rpn_feature']=a;batch['spatial_features_2d']=c[:,:32,0]+c[:,32:,0]+torch.nn.functional.interpolate(a,size=l.shape[2:],mode='bilinear',align_corners=False)
  return batch
class Head(nn.Module):
 def __init__(self):
  super().__init__();self.scale=nn.Parameter(torch.ones(()));self.forward_ret_dict={}
  self.model_cfg=SimpleNamespace(LOSS_CONFIG=SimpleNamespace(LOSS_WEIGHTS={'iou_weight':2.}))
 def forward(self,batch):
  self.forward_ret_dict.clear();self.forward_ret_dict['feature']=batch['spatial_features_2d']*self.scale
  gt=batch['gt_boxes'];self.forward_ret_dict['empty']=gt.shape[1]==0
  self.forward_ret_dict['target']=gt.mean() if gt.numel() else gt.new_zeros(())
  self.forward_ret_dict['box_cls_labels']=torch.zeros(1,4,dtype=torch.long) if not gt.numel() else torch.ones(1,4,dtype=torch.long)
  return batch
 def get_loss(self):
  error=self.forward_ret_dict['feature']-self.forward_ret_dict['target'];cls=error.square().mean();loc=.5*error.abs().mean()
  direction=.2*error.mean().square();iou=.1*error.square().mean()
  if self.forward_ret_dict['empty']:loc=loc*0;direction=direction*0;iou=iou*0
  loss=cls+loc+direction+2*iou
  return loss,dict(rpn_loss_cls=float(cls.detach()),rpn_loss_loc=float(loc.detach()),rpn_loss_dir=float(direction.detach()),rpn_loss_iou=float(iou.detach()),rpn_loss=float(loss.detach()))
class Model(nn.Module):
 def __init__(self):
  super().__init__();self.backbone_3d=Backbone();self.map_to_bev_module=nn.Identity();self.backbone_2d=nn.Identity();self.dense_head=Head()
  self.lidar_model=nn.Identity();self.dense_head_2d=nn.Identity();self.depth_loss_head=nn.Identity();self.register_buffer('global_step',torch.tensor(13))
def inputs():return dict(batch_size=1,left_img=torch.randn(1,3,16,32),right_img=torch.randn(1,3,16,32),frame_id=['001'],calib=[None],image_shape=[[16,32]])

class Tests(unittest.TestCase):
 def test_native_empty_GT_keeps_background_objective_and_receiver_gradient(self):
  model=Model();selected=freeze_except_codec(model);observer=NativeTaskAdaptation(model)
  try:
   loss=observer.forward(inputs(),torch.empty(1,0,8));loss.backward();row=observer.finish_backward()
   self.assertTrue(torch.isfinite(loss));self.assertTrue(row['empty_GT']);self.assertEqual(row['GT_shape'],[1,0,8])
   self.assertEqual(row['positive_anchors'],0);self.assertEqual(row['background_anchors'],4)
   self.assertTrue(all(row[k]==0 for k in ('location_loss','direction_loss','iou_loss_raw')))
   self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for _,p in selected))
   self.assertTrue(all(v>0 for v in row['gradient_norms'].values()));self.assertEqual(observer.calls['forbidden'],0)
  finally:observer.close()
 def test_actual_16_optimizer_step_and_native_receiver_backward(self):
  torch.manual_seed(17);model=Model();selected=freeze_except_codec(model);reference={k:v.clone() for k,v in model.state_dict().items()};observer=NativeTaskAdaptation(model)
  try:
   loss=observer.forward(inputs(),torch.ones(1,1,8));loss.backward()
   self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for _,p in selected))
   self.assertTrue(all(p.grad is None for n,p in model.named_parameters() if n not in {name for name,_ in selected}))
   row=observer.finish_backward();self.assertEqual(row['sequence'],SEQUENCE);self.assertEqual(set(row['gradient_norms']),{'left_stereo','right_stereo','appearance','native_cost','symbols'})
   self.assertTrue(all(v>0 for v in row['gradient_norms'].values()));self.assertTrue(row['GT_introduced_only_at_head'])
   optimizer=torch.optim.AdamW([p for _,p in selected],lr=.0001);optimizer.step()
   self.assertEqual(len(optimizer.state),16);self.assertTrue(all(float(s['step'])==1 for s in optimizer.state.values()))
   self.assertEqual(assert_frozen(reference,model.state_dict(),{n for n,_ in selected}),len(reference)-16)
   self.assertEqual(observer.calls['forbidden'],0);self.assertFalse(model.dense_head.forward_ret_dict)
  finally:observer.close()
 def test_GT_contents_cannot_change_sender_or_received_features(self):
  torch.manual_seed(18);model=Model();freeze_except_codec(model);observer=NativeTaskAdaptation(model);sensors=inputs();received=[]
  hook=model.backbone_3d.stereo_feature_link.register_forward_hook(lambda m,i,o:received.append(tuple(x.detach().clone() for x in o)))
  try:
   for value in (1.,7.):
    loss=observer.forward(sensors,torch.full((1,1,8),value));loss.backward();observer.finish_backward()
   self.assertTrue(all(torch.equal(a,b) for a,b in zip(received[0],received[1])))
  finally:hook.remove();observer.close()
 def test_non_sensor_target_train_mode_teacher_and_broken_cost_graph_rejected(self):
  model=Model();freeze_except_codec(model);observer=NativeTaskAdaptation(model)
  try:
   batch=inputs();batch['gt_boxes']=torch.ones(1,1,8)
   with self.assertRaises(ValueError):observer.forward(batch,torch.ones(1,1,8))
   model.dense_head.train()
   with self.assertRaises(ValueError):observer.forward(inputs(),torch.ones(1,1,8))
   model.dense_head.eval()
   with self.assertRaises(RuntimeError):model.lidar_model({})
   model.backbone_3d.build_cost.detach=True
   with self.assertRaisesRegex(RuntimeError,'lacks gradient'):observer.forward(inputs(),torch.ones(1,1,8))
  finally:observer.close()
 def test_independent_loss_weight_target_barrier_and_cost_gradient_audit(self):
  shapes={'left_stereo':[1,32,320,1248],'right_stereo':[1,32,320,1248],'appearance':[1,32,80,312],'symbols':[1,62400,2],'native_cost':[1,64,72,80,312]}
  row=dict(step=1,epoch=1,frame_id='001',loss=8.,classification_loss=1.,location_loss=2.,direction_loss=3.,iou_loss_raw=1.,iou_weight=2.,native_reported_loss=8.,
   preclip_grad_norm=1.,tx_energy=62400.,lr=.0001,cbr=1/38.4,channel='identity',allocation='uniform',snr_db=10.,data_complex_uses=62400,total_complex_uses=62400,
   pilot_complex_uses=0,header_complex_uses=0,boundary='stereo_features_before_receiver_cost',stereo_complex_uses=49920,appearance_complex_uses=12480,
   feature_shapes={k:v for k,v in shapes.items() if k in ('left_stereo','right_stereo','appearance')},gradient_shapes=shapes,gradient_norms={k:1. for k in shapes},
   sequence=SEQUENCE,cost_inputs_are_received_features=True,appearance_input_is_received_feature=True,GT_introduced_only_at_head=True,all_modules_eval=True,
   GT_shape=[1,2,8],empty_GT=False,positive_anchors=3,background_anchors=5,
   sender_input_keys=['batch_size','left_img','right_img','calib','image_shape','frame_id'],random_T_present=False,loss_type='native3D_cls_box_direction')
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/'records.jsonl';path.write_text(json.dumps(row)+'\n');self.assertEqual(audit.scalar_records(path,{'001'},1,2.)['steps'],1)
   for key,value in [('iou_weight',1.),('loss',7.),('GT_introduced_only_at_head',False),('sender_input_keys',row['sender_input_keys']+['gt_boxes']),('gradient_norms',{**row['gradient_norms'],'native_cost':0.})]:
    changed=dict(row);changed[key]=value;path.write_text(json.dumps(changed)+'\n')
    with self.assertRaises(ValueError):audit.scalar_records(path,{'001'},1,2.)
   empty=dict(row,GT_shape=[1,0,8],empty_GT=True,positive_anchors=0,background_anchors=8,
    loss=1.,native_reported_loss=1.,location_loss=0.,direction_loss=0.,iou_loss_raw=0.)
   path.write_text(json.dumps(empty)+'\n');self.assertEqual(audit.scalar_records(path,{'001'},1,2.)['empty_GT_frames'],1)
   for changed in [dict(empty,positive_anchors=1),dict(empty,background_anchors=0),dict(empty,empty_GT=False),
                   dict(empty,location_loss=1.,loss=2.,native_reported_loss=2.)]:
    path.write_text(json.dumps(changed)+'\n')
    with self.assertRaises(ValueError):audit.scalar_records(path,{'001'},1,2.)

if __name__=='__main__':unittest.main()
