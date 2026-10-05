import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from geocomm.stereo_feature_link import StereoFeatureLink
from geocomm.stereo_feature_warmup import freeze_except_codec,capture_reconstruction,assert_frozen
from geocomm.stereo_feature_observer import StereoFeatureObserver
from diagnose_codec_features import compare_features
spec=importlib.util.spec_from_file_location('audit_F5',ROOT/'scripts/audit_stereo_feature_warmup.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

class Student(nn.Module):
    def __init__(self):
        super().__init__();self.scale=nn.Parameter(torch.ones(()))
    def forward(self,value): return value*self.scale,torch.nn.functional.avg_pool2d(value,4)*self.scale+1
class Cost(nn.Module):
    def forward(self,*inputs): return None
class Backbone(nn.Module):
    def __init__(self):
        super().__init__();self.semantic_link=None;self.student_semantic_link_encoder=Student()
        self.stereo_feature_link=StereoFeatureLink(channel='identity');self.build_cost=Cost()
        self.feature_backbone=nn.Identity();self.feature_neck=nn.Identity();self.after_link=0;self.bypass=False
    def forward(self,batch):
        left,app=self.student_semantic_link_encoder(batch['left_img'])
        right,_=self.student_semantic_link_encoder(batch['right_img'])
        received=self.stereo_feature_link(left,right,app,batch);self.after_link+=1
        l,r=received[:2] if not self.bypass else (left,right)
        self.build_cost(l,r,None,None,None)
        batch['rpn_feature']=received[2];return batch
class Model(nn.Module):
    def __init__(self):
        super().__init__();self.backbone_3d=Backbone();self.lidar_model=nn.Identity();self.head=nn.Linear(2,2)

def batch():
    return dict(batch_size=1,left_img=torch.randn(1,32,8,12),right_img=torch.randn(1,32,8,12),frame_id=['001'])

class Tests(unittest.TestCase):
    def test_exact_frozen_scope_actual_step_and_stop_before_cost(self):
        torch.manual_seed(17);model=Model();selected=freeze_except_codec(model)
        self.assertEqual(len(selected),16);reference={k:v.clone() for k,v in model.state_dict().items()}
        sensors=batch();b=model.backbone_3d
        left,app=b.student_semantic_link_encoder(sensors['left_img']);right,_=b.student_semantic_link_encoder(sensors['right_img'])
        received=b.stereo_feature_link(left,right,app,dict(sensors))
        expected=.5*((received[0]-left).square().mean()+(received[1]-right).square().mean())+(received[2]-app).square().mean()
        loss,scalars=capture_reconstruction(b,sensors)
        self.assertTrue(torch.equal(loss,expected));self.assertEqual(b.after_link,0)
        loss.backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for _,p in selected))
        self.assertIsNone(b.student_semantic_link_encoder.scale.grad)
        optimizer=torch.optim.AdamW([p for _,p in selected],lr=.0001);optimizer.step()
        self.assertTrue(all(float(state['step'])==1 for state in optimizer.state.values()))
        names={name for name,_ in selected}
        self.assertEqual(assert_frozen(reference,model.state_dict(),names),len(reference)-16)
        self.assertEqual(audit.inactive_states(reference,{k:v.detach() for k,v in model.state_dict().items()},names)[0],len(reference)-16)
        sensors['gt_boxes']=torch.ones(1)
        with self.assertRaises(ValueError): capture_reconstruction(b,sensors)
        self.assertEqual(len(b.stereo_feature_link._forward_hooks),0)
    def test_passive_receiver_sequence_and_clean_bypass_rejection(self):
        model=Model().eval();rows=[];observer=StereoFeatureObserver(model,'identity',rows.append,compare_features)
        try:
            with torch.no_grad(): model.backbone_3d(batch())
            self.assertEqual(observer.calls,dict(frames=1,student=2,codec=1,channel=1,build_cost=1,forbidden=0))
            self.assertEqual(rows[0]['sequence'],['student','student','link_start','channel','link_done','build_cost'])
            model.backbone_3d.bypass=True
            with self.assertRaisesRegex(RuntimeError,'bypass'):
                with torch.no_grad(): model.backbone_3d(batch())
        finally: observer.close()
    def test_independent_scalar_audit_rejects_wrong_wire_loss_and_shapes(self):
        row=dict(step=1,epoch=1,frame_id='001',loss=5.,left_stereo_mse=2.,right_stereo_mse=4.,appearance_mse=2.,
            preclip_grad_norm=1.,tx_energy=62400.,lr=.0001,cbr=1/38.4,channel='identity',allocation='uniform',
            data_complex_uses=62400,total_complex_uses=62400,pilot_complex_uses=0,header_complex_uses=0,snr_db=10.,
            boundary='stereo_features_before_receiver_cost',stereo_complex_uses=49920,appearance_complex_uses=12480,
            **{branch+'_shape':([1,32,80,312] if branch=='appearance' else [1,32,320,1248]) for branch in ('left_stereo','right_stereo','appearance')})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'records.jsonl';path.write_text(json.dumps(row)+'\n')
            self.assertEqual(audit.scalar_records(path,{'001'},1)['steps'],1)
            for field,value in [('loss',6.),('stereo_complex_uses',49921),('right_stereo_shape',[1,32,80,311]),('frame_id','002')]:
                changed=dict(row);changed[field]=value;path.write_text(json.dumps(changed)+'\n')
                with self.assertRaises(ValueError): audit.scalar_records(path,{'001'},1)

if __name__=='__main__': unittest.main()
