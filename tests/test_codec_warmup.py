import importlib.util
from pathlib import Path
import sys
import unittest
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from geocomm.link import GeometryLink
from geocomm.codec_warmup import capture_reconstruction,freeze_except_codec,assert_frozen
spec=importlib.util.spec_from_file_location('F4_audit',ROOT/'scripts/audit_codec_warmup.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.semantic_link=GeometryLink(4,4,2,channel='identity',allocation='uniform',task_sensitivity=True)
        self.semantic_link_boundary='raw_cost'
        self.register_buffer('cost',torch.randn(1,4,6,8,10))
        self.register_buffer('appearance',torch.randn(1,4,16,20))
        self.register_buffer('global_step',torch.tensor(13))
        self.downstream_calls=0

    def forward(self,batch):
        result=self.semantic_link(self.cost,self.appearance,batch,torch.arange(6.))
        self.downstream_calls+=1
        return result


class Model(nn.Module):
    def __init__(self):
        super().__init__();self.backbone_3d=Backbone();self.head=nn.Linear(2,2)


class CodecWarmupTests(unittest.TestCase):
    def test_actual_optimizer_update_and_stop_before_receiver(self):
        torch.manual_seed(17);model=Model();selected=freeze_except_codec(model)
        self.assertEqual(len(selected),16)
        reference={k:v.clone() for k,v in model.state_dict().items()}
        batch=dict(batch_size=1,left_img=torch.zeros(1,3,32,40),frame_id=['001'])
        outputs=model.backbone_3d.semantic_link(model.backbone_3d.cost,model.backbone_3d.appearance,dict(batch),torch.arange(6.))
        expected=sum((a-b).square().mean() for a,b in zip(outputs,(model.backbone_3d.cost,model.backbone_3d.appearance)))
        loss,scalars=capture_reconstruction(model.backbone_3d,batch)
        self.assertTrue(torch.equal(loss,expected));self.assertEqual(model.backbone_3d.downstream_calls,0)
        optimizer=torch.optim.AdamW([p for _,p in selected],lr=.0001)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for _,p in selected))
        optimizer.step()
        names={name for name,_ in selected}
        count=assert_frozen(reference,model.state_dict(),names)
        self.assertEqual(count,len(reference)-16)
        self.assertEqual(scalars['channel'],'identity')
        self.assertTrue(all(float(state['step'])==1 for state in optimizer.state.values()))
        self.assertEqual(len(model.backbone_3d.semantic_link._forward_hooks),0)

    def test_inactive_posterior_sensitivity_and_step_mutations_rejected(self):
        model=Model();names={name for name,_ in freeze_except_codec(model)}
        reference={k:v.clone() for k,v in model.state_dict().items()}
        for key in ['backbone_3d.global_step','backbone_3d.semantic_link.posterior.weight',
                    'backbone_3d.semantic_link.sensitivity_predictor.cost.0.weight','head.weight']:
            changed={k:v.clone() for k,v in reference.items()};changed[key]+=1
            with self.assertRaises(ValueError): audit.inactive_states(reference,changed,names)
        lossless={k:v.clone() for k,v in reference.items()}
        self.assertEqual(audit.inactive_states(reference,lossless,names)[0],len(reference)-16)

    def test_non_sensor_input_and_live_target_gradients_rejected(self):
        model=Model();freeze_except_codec(model)
        batch=dict(batch_size=1,left_img=torch.zeros(1,3,32,40),frame_id=['001'],depth_gt_img=torch.zeros(1))
        with self.assertRaises(ValueError): capture_reconstruction(model.backbone_3d,batch)
        del batch['depth_gt_img'];model.backbone_3d.cost.requires_grad_(True)
        with self.assertRaises(RuntimeError): capture_reconstruction(model.backbone_3d,batch)
        self.assertEqual(len(model.backbone_3d.semantic_link._forward_hooks),0)


if __name__=='__main__': unittest.main()
