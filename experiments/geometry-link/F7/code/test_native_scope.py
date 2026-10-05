"""CPU regression fixtures only; no native GPU/F7 performance results."""
import copy,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path[:0]=[str(HERE),str(ROOT/'src'),str(ROOT/'scripts')]
import conditions as c
import audit as a
from native_observer import MatchedNativeTaskAdaptation,freeze_matched_codec
from geocomm.stereo_task_adaptation import NativeTaskAdaptation,freeze_except_codec
spec=importlib.util.spec_from_file_location('F6_fixture',ROOT/'tests/test_stereo_task_adaptation.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
spec=importlib.util.spec_from_file_location('F7_fixture',HERE/'test_conditions.py')
f7=importlib.util.module_from_spec(spec);spec.loader.exec_module(f7)

class Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_explicit_noisy_scope_keeps_GT_barrier_teacher_rejection_and_all_codec_gradients(self):
        model=fixture.Model();model.backbone_3d.stereo_feature_link.channel.kind='awgn';selected=freeze_matched_codec(model,'awgn')
        with self.assertRaises(ValueError):freeze_except_codec(model)
        with self.assertRaises(ValueError):NativeTaskAdaptation(model)
        with self.assertRaises(ValueError):MatchedNativeTaskAdaptation(model,'identity')
        observer=MatchedNativeTaskAdaptation(model,'awgn');batch=fixture.inputs();outputs=[]
        hook=model.backbone_3d.stereo_feature_link.register_forward_hook(lambda m,i,o:outputs.append(tuple(v.detach().clone() for v in o)))
        try:
            rng=torch.get_rng_state().clone()
            for gt in (torch.ones(1,1,8),torch.full((1,1,8),7.)):
                torch.set_rng_state(rng);model.zero_grad(set_to_none=True);loss=observer.forward(batch,gt);loss.backward();row=observer.finish_backward()
                self.assertEqual(row['channel'],'awgn');self.assertEqual(row['sequence'],a.SEQUENCE)
                self.assertTrue(row['GT_introduced_only_at_head']);self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for _,v in selected))
                self.assertTrue(all(v>0 for v in row['gradient_norms'].values()))
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(*outputs)))
            with self.assertRaises(RuntimeError):model.lidar_model({})
            batch['gt_boxes']=torch.ones(1,1,8)
            with self.assertRaises(ValueError):observer.forward(batch,torch.ones(1,1,8))
        finally:hook.remove();observer.close()

    def test_actual_16_Adam_moments_steps_and_corruption_rejected(self):
        names=sorted(a.expected_names());parameters=[torch.nn.Parameter(torch.ones(2,3)) for _ in names]
        optimizer=torch.optim.AdamW([dict(params=parameters,parameter_names=names)],lr=.0001,weight_decay=.0001)
        for _ in range(6):
            optimizer.zero_grad(set_to_none=True);sum(v.square().mean() for v in parameters).backward();optimizer.step()
        saved=dict(model_state=dict(zip(names,[v.detach().clone() for v in parameters])),optimizer_state=optimizer.state_dict())
        a.audit_optimizer(saved,names,6)
        for field in ('step','moment','scope','settings'):
            changed=copy.deepcopy(saved);opt=changed['optimizer_state'];first=next(iter(opt['state']))
            if field=='step':opt['state'][first]['step']=torch.tensor(5.)
            elif field=='moment':opt['state'][first]['exp_avg'][0,0]=float('nan')
            elif field=='scope':opt['state'].pop(first)
            else:opt['param_groups'][0]['lr']=.001
            with self.assertRaises(ValueError):a.audit_optimizer(changed,names,6)

    def test_independent_native_record_audit_rejects_wrong_loss_barrier_symbols_SNR_and_GT(self):
        # Native F6b scalar examples are used solely as test fixtures. Their
        # channel/augmentation metadata below is synthetic, never F7 evidence.
        source=ROOT/'data/runs/stereo-native-task-seed17-002-training-audit.records/training.jsonl'
        rows=[json.loads(line) for line in source.read_text().splitlines()[:6]]
        from geocomm.stereo_feature_link import StereoFeatureLink
        link=StereoFeatureLink(channel='awgn').eval();controller=c.PairedChannel(link,'awgn','cpu')
        try:
            for index,row in enumerate(rows,1):
                batch,gt=f7.sensors(frame=row['frame_id'],empty=row['empty_GT'])
                batch['left_img']=np.ones((1,3,320,1248),dtype=np.float32);batch['right_img']=batch['left_img'].copy()
                gt=np.zeros(row['GT_shape'],dtype=np.float32)
                if not row['random_T_present']:batch.pop('random_T')
                snr=controller.prepare(index);link.channel(torch.ones(1,62400,2),snr)
                row.update(channel='awgn',snr_db=snr,channel_condition=copy.deepcopy(controller.last_evidence),augmented_evidence=c.augmented_evidence(batch,gt))
            saved=controller.checkpoint_rng()
        finally:controller.close()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'fixture.jsonl'
            def write(values):path.write_text(''.join(json.dumps(v)+'\n' for v in values))
            write(rows);_,summary=a.scalar_records(path,{r['frame_id'] for r in rows},6,rows[0]['iou_weight'],'awgn')
            self.assertGreater(summary['empty_GT_frames'],0)
            self.assertEqual(c.replay_rng_checkpoint(saved,'awgn',6,rows)['state'],'passed')
            corrupt=copy.deepcopy(saved);corrupt['final_state'][0]^=1
            with self.assertRaises(ValueError):c.replay_rng_checkpoint(corrupt,'awgn',6,rows)
            for field,value in [('loss',rows[0]['loss']+1),('GT_introduced_only_at_head',False),('total_complex_uses',62401),
                                ('snr_db',10.),('sender_input_keys',rows[0]['sender_input_keys']+['gt_boxes'])]:
                changed=copy.deepcopy(rows);changed[0][field]=value;write(changed)
                with self.assertRaises(ValueError):a.scalar_records(path,{r['frame_id'] for r in rows},6,rows[0]['iou_weight'],'awgn')

if __name__=='__main__':unittest.main()
