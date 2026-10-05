"""Counter/energy/identity corruption rejection fixtures for staged auditing."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import torch
from PIL import Image
from audit_training import accounting,optimizer_audit,policy
from data import StereoRGB,digest


class AuditTests(unittest.TestCase):
    def test_unfreeze_and_fresh_adam_counts(self):
        names=['semantic.global_encoder.weight','semantic.global_decoder.flow.weight']
        counts={names[0]:3,names[1]:1}
        groups=[];states={};model={name:torch.zeros(2) for name in names}
        rates,_=policy(1,12)
        for index,key in enumerate(['global_encoder','global_decoder','flow','fusions']):
            matched=[name for name in names if ('flow' if '.flow.' in name else 'global_encoder')==key]
            ids=[index] if matched else []
            groups.append(dict(component=key,parameter_names=matched,params=ids,lr=rates[key],weight_decay=0.,betas=(.9,.999),eps=1e-8,amsgrad=False))
            if matched: states[index]=dict(step=torch.tensor(float(counts[matched[0]])),exp_avg=torch.zeros(2),exp_avg_sq=torch.ones(2))
        saved=dict(model_state=model,optimizer_state=dict(param_groups=groups,state=states))
        self.assertEqual(optimizer_audit(saved,names,counts,1),2)
        broken=copy.deepcopy(saved);broken['optimizer_state']['state'][2]['step']=torch.tensor(3.)
        with self.assertRaisesRegex(ValueError,'Adam steps'): optimizer_audit(broken,names,counts,1)
        broken=copy.deepcopy(saved);broken['optimizer_state']['state'][0]['exp_avg_sq'][0]=-1.
        with self.assertRaisesRegex(ValueError,'negative second'): optimizer_audit(broken,names,counts,1)

    def test_wire_odd_padding_overlap_and_control(self):
        row={'views':[dict(shape=[197,203],boxes=[dict(xyxy=[1,1,4,4]),dict(xyxy=[2,2,4,4])]),dict(shape=[197,203],boxes=[])]}
        # Padding198x204; four support cells in union, global2*33*34*9.
        real=9*(2*33*34+4);data=(real+1)//2;control=1176+2*672
        value=dict(key_cells=[4,0],data_real_values=real,padding_real_values=real%2,data_uses=data,
            control_uses=control,pilot_uses=0,total_uses=data+control,total_energy=data+control,
            cbr_complex_per_rgb_real_value=(data+control)/(6*197*203))
        accounting(value,row)
        for field in ('control_uses','data_uses','pilot_uses'):
            broken=copy.deepcopy(value);broken[field]+=1
            with self.assertRaises(ValueError): accounting(broken,row)
        broken=copy.deepcopy(value);broken['total_energy']*=.99
        with self.assertRaisesRegex(ValueError,'energy'): accounting(broken,row)

    def test_loader_rejects_image_and_fold_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);views=[]
            for camera in ['image_2','image_3']:
                directory=root/'training'/camera;directory.mkdir(parents=True)
                image=directory/'000000.png';Image.new('RGB',(203,197)).save(image)
                views.append(dict(camera=camera,shape=[197,203],image_sha256=digest(image),boxes=[]))
            records=root/'records.jsonl';records.write_text(json.dumps(dict(frame_id='000000',views=views))+'\n')
            fold=root/'fold.json';fold.write_text(json.dumps(dict(folds={'x':dict(ids=['000000'])})))
            audit=root/'audit.json';audit.write_text(json.dumps(dict(state='passed',records_sha256=digest(records),fold_sha256=digest(fold),split='x',frames=1)))
            dataset=StereoRGB(root,records,audit,fold,'x');self.assertEqual(list(dataset[0]['left'].shape),[1,3,197,203])
            (root/'training/image_2/000000.png').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'image identity'): dataset[0]
            fold.write_text(fold.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'ROI/fold'): StereoRGB(root,records,audit,fold,'x')


if __name__=='__main__': unittest.main()
