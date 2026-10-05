import copy
import unittest
import torch
from torch import nn
from geocomm.pooling_diagnostic import CONDITIONS,PoolingDiagnostic,pool_restore,state_hashes


class Student(nn.Module):
    def forward(self,value): return value,value+1

class Backbone(nn.Module):
    def __init__(self):
        super().__init__();self.semantic_link=None;self.student_semantic_link_encoder=Student()
        self.dres0=nn.Identity();self.feature_backbone=nn.Identity();self.feature_neck=nn.Identity()
    def forward(self,batch):
        stereo,app=self.student_semantic_link_encoder(batch['left_img'])
        right,_=self.student_semantic_link_encoder(batch['right_img'])
        return self.dres0(torch.stack((stereo,right),dim=2)),app

class Model(nn.Module):
    def __init__(self):
        super().__init__();self.backbone_3d=Backbone();self.lidar_model=nn.Identity()

class Tests(unittest.TestCase):
    def test_all_interventions_preserve_stereo_and_state(self):
        left=torch.arange(1*2*9*11).float().reshape(1,2,9,11)/13;right=left.flip(-1)
        batch=dict(batch_size=1,frame_id=['000000'],left_img=left,right_img=right)
        for condition in CONDITIONS:
            model=Model().eval();rows=[];before=state_hashes(model)
            recorder=PoolingDiagnostic(model,condition,rows.append)
            with torch.no_grad(): cost,app=model.backbone_3d(batch)
            recorder.close()
            original=torch.stack((left,right),dim=2)
            expected_cost=pool_restore(original,(1 if condition=='depth_preserved' else 8,4,4))[0] if condition in ('both','cost_only','depth_preserved') else original
            expected_app=pool_restore(left+1,(4,4))[0] if condition in ('both','appearance_only') else left+1
            self.assertTrue(torch.equal(cost,expected_cost));self.assertTrue(torch.equal(app,expected_app))
            self.assertEqual(state_hashes(model),before);self.assertEqual(rows[0]['student_calls'],2)
            self.assertEqual(recorder.calls,dict(frames=1,student=2,raw_cost=1,forbidden=0))
            self.assertEqual(rows[0]['appearance_pooled_shape'],[1,2,3,3] if condition in ('both','appearance_only') else None)
    def test_depth_axis_preserved_and_control_returns_same_objects(self):
        value=torch.arange(9).float().reshape(1,1,9,1,1).expand(1,1,9,7,11)
        preserved,shape=pool_restore(value,(1,4,4));self.assertTrue(torch.allclose(value,preserved))
        coarse,_=pool_restore(value,(8,4,4));self.assertFalse(torch.allclose(value,coarse))
        model=Model().eval();record=PoolingDiagnostic(model,'control',lambda row:None)
        batch=dict(batch_size=1,frame_id=['0'],left_img=torch.ones(1,1,4,4),right_img=torch.zeros(1,1,4,4))
        with torch.no_grad(): record.start(model.backbone_3d,(batch,))
        outputs=(batch['left_img'],batch['left_img'])
        self.assertIsNone(record.student(model.backbone_3d.student_semantic_link_encoder,(batch['left_img'],),outputs));record.close()
    def test_forbidden_inputs_and_teachers_raise(self):
        model=Model().eval();r=PoolingDiagnostic(model,'control',lambda row:None)
        with self.assertRaisesRegex(RuntimeError,'autograd'):
            model.backbone_3d(dict(batch_size=1,frame_id=['0']))
        r.close()
        for bad in ['gt_boxes','depth_maps','points']:
            model=Model().eval();r=PoolingDiagnostic(model,'both',lambda row:None)
            with self.assertRaisesRegex(RuntimeError,'non-sensor'): model.backbone_3d(dict(batch_size=1,frame_id=['0'],**{bad:torch.zeros(1)}))
            r.close()
        model=Model().eval();r=PoolingDiagnostic(model,'control',lambda row:None)
        with self.assertRaisesRegex(RuntimeError,'teacher'): model.lidar_model(torch.zeros(1))
        r.close()


# Independent corruption fixtures for the report audit and native weight loader.
class EvidenceTests(unittest.TestCase):
    def test_operation_metadata_rejects_wrong_condition_grid_or_inputs(self):
        import importlib.util
        from pathlib import Path
        spec=importlib.util.spec_from_file_location('audit_pooling',Path(__file__).resolve().parents[1]/'scripts/audit_pooling.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        row=dict(frame_id='000000',condition='depth_preserved',communication_enabled=False,autograd_enabled=False,student_calls=2,raw_cost_calls=1,
            sensor_input_keys=['batch_size','left_img','right_img','calib','image_shape','frame_id'],
            cost_shape=[1,64,72,80,312],appearance_shape=[1,32,80,312],
            cost_operation='adaptive_avg_pool_1_4_4_trilinear_restore',appearance_operation='identity',
            cost_pooled_shape=[1,64,72,20,78],appearance_pooled_shape=None)
        module.audit_rows([row],'depth_preserved',['000000'])
        for key,value in [('autograd_enabled',True),('cost_pooled_shape',[1,64,9,20,78]),('condition','both'),('raw_cost_calls',2),('sensor_input_keys',row['sensor_input_keys']+['gt_boxes'])]:
            bad=copy.deepcopy(row);bad[key]=value
            with self.assertRaises(ValueError): module.audit_rows([bad],'depth_preserved',['000000'])
        with self.assertRaises(ValueError): module.audit_rows([row],'depth_preserved',['000001'])

    def test_loader_full_coverage_and_native_silent_skip_rejected(self):
        import importlib.util
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        spec=importlib.util.spec_from_file_location('diagnose_pooling',Path(__file__).resolve().parents[1]/'scripts/diagnose_pooling.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        model=nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(519)])
        state={name:torch.ones_like(value) for name,value in model.state_dict().items()}
        state.update({name:torch.zeros(1) for name in module.unused_link_states()})
        self.assertEqual(len(module.unused_link_states()),30)
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.pth';torch.save(dict(model_state=state),path)
            with self.assertRaisesRegex(ValueError,'fixed F4'): module.frozen_load(model,path)
            with patch.object(module,'sha256',return_value=module.CHECKPOINT_SHA):
                with self.assertRaisesRegex(RuntimeError,'native loader'): module.frozen_load(model,path,loader=lambda **kw:None)
                loaded=module.frozen_load(model,path);self.assertEqual(loaded['required_states'],519)
                self.assertTrue(any(parameter.requires_grad for parameter in model.parameters()))
                del state['0'];torch.save(dict(model_state=state),path)
                with self.assertRaisesRegex(ValueError,'complete519'): module.frozen_load(model,path)
                state['0']=torch.tensor([float('nan')]);torch.save(dict(model_state=state),path)
                with self.assertRaisesRegex(ValueError,'finite'): module.frozen_load(model,path)

if __name__=='__main__': unittest.main()
