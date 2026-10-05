"""CPU regression gates for F8 scope, native barriers and isolated paired noise."""
import argparse,copy,hashlib,importlib.util,json,os,sys,tempfile,unittest
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];sys.path[:0]=[str(HERE),str(ROOT/'src')]
from native_observer import freeze_scope,ScopedNativeTaskAdaptation,student_names
from geocomm.student import StereoStudentEncoder
from geocomm.stereo_task_adaptation import assert_frozen
from geocomm.stereo_feature_link import StereoFeatureLink
import conditions as c
import audit as a
spec=importlib.util.spec_from_file_location('native_fixture',ROOT/'tests/test_stereo_task_adaptation.py');fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
def model():
    m=fixture.Model();m.backbone_3d.student_semantic_link_encoder=StereoStudentEncoder();m.backbone_3d.stereo_feature_link.channel.kind='awgn';return m
class Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(17)
    def test_eligibility_project_root_matches_runner(self):
        import eligibility
        self.assertEqual(eligibility.ROOT,ROOT)
        self.assertTrue((eligibility.ROOT/'experiments/geometry-link/F8-matched-encoder-adaptation.md').is_file())
    def test_both_scopes_actual_gradients_Adam_and_frozen_state(self):
        for scope,count in [('codec',16),('joint',51)]:
            m=model();selected=freeze_scope(m,scope);self.assertEqual(len(selected),count)
            names={n for n,_ in selected};reference={n:v.clone() for n,v in m.state_dict().items()};o=ScopedNativeTaskAdaptation(m,scope)
            opt=torch.optim.AdamW([dict(params=[v for _,v in selected],parameter_names=[n for n,_ in selected])],lr=.0001,weight_decay=.0001)
            try:
                for target in [torch.ones(1,1,8),torch.empty(1,0,8)]:
                    opt.zero_grad(set_to_none=True);loss=o.forward(fixture.inputs(),target);loss.backward()
                    self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() and torch.count_nonzero(v.grad)>0 for _,v in selected))
                    self.assertTrue(all(v.grad is None for n,v in m.named_parameters() if n not in names));row=o.finish_backward();self.assertTrue(row['GT_introduced_only_at_head'])
                    opt.step()
                self.assertEqual(assert_frozen(reference,m.state_dict(),names),len(reference)-count)
                saved=dict(model_state=m.state_dict(),optimizer_state=opt.state_dict());a.audit_optimizer(saved,[n for n,_ in selected],2)
                corrupted=copy.deepcopy(saved);next(iter(corrupted['optimizer_state']['state'].values()))['step']=torch.tensor(1.)
                with self.assertRaises(ValueError):a.audit_optimizer(corrupted,[n for n,_ in selected],2)
            finally:o.close()
    def test_scope_rejects_changed_architecture_and_channel(self):
        m=model()
        with self.assertRaises(ValueError):freeze_scope(m,'identity')
        m.backbone_3d.stereo_feature_link.channel.kind='identity'
        with self.assertRaises(ValueError):freeze_scope(m,'joint')
        m=model();m.backbone_3d.student_semantic_link_encoder.register_parameter('extra',torch.nn.Parameter(torch.ones(())))
        with self.assertRaises(ValueError):freeze_scope(m,'joint')
    def test_GT_changes_cannot_reach_sender_and_forbidden_modules_rejected(self):
        m=model();freeze_scope(m,'joint');o=ScopedNativeTaskAdaptation(m,'joint');batch=fixture.inputs();received=[]
        h=m.backbone_3d.stereo_feature_link.register_forward_hook(lambda m,i,out:received.append(tuple(v.detach().clone() for v in out)))
        rng=torch.get_rng_state().clone()
        try:
            for gt in (torch.ones(1,1,8),torch.full((1,1,8),7.)):
                torch.set_rng_state(rng);m.zero_grad(set_to_none=True);o.forward(batch,gt).backward();o.finish_backward()
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(*received)))
            with self.assertRaises(RuntimeError):m.lidar_model({})
            with self.assertRaises(ValueError):o.forward(dict(batch,gt_boxes=torch.ones(1,1,8)),torch.ones(1,1,8))
        finally:h.remove();o.close()
    def test_joint_detached_student_output_rejected(self):
        m=model();freeze_scope(m,'joint');o=ScopedNativeTaskAdaptation(m,'joint')
        try:
            with self.assertRaises(RuntimeError):o.student(m.backbone_3d.student_semantic_link_encoder,(),(torch.ones(1),torch.ones(1)))
        finally:o.close()
    def test_dedicated_noise_replay_pairing_and_global_RNG_unchanged(self):
        pairs=[];saved=[]
        for scope in ('codec','joint'):
            link=StereoFeatureLink(channel='awgn').eval();controller=c.PairedChannel(link,'awgn','cpu');rows=[];outputs=[]
            try:
                state=torch.get_rng_state().clone()
                for i in range(1,7):
                    snr=controller.prepare(i);out=link.channel(torch.ones(1,62400,2),snr);outputs.append(out[0].clone());rows.append(dict(channel_condition=copy.deepcopy(controller.last_evidence)))
                self.assertTrue(torch.equal(state,torch.get_rng_state()));saved.append(controller.checkpoint_rng());pairs.append(outputs)
                self.assertEqual(c.replay_rng_checkpoint(saved[-1],'awgn',6,rows)['state'],'passed')
                with self.assertRaises(ValueError):controller.prepare(6)
            finally:controller.close()
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(*pairs)));self.assertTrue(torch.equal(saved[0]['final_state'],saved[1]['final_state']))
    def test_pair_audit_detects_input_and_noise_mismatch(self):
        spec=importlib.util.spec_from_file_location('sensor_fixture',ROOT/'experiments/geometry-link/F7/code/test_conditions.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        arms=[]
        for scope in ('codec','joint'):
            controller=c.PairedChannel(StereoFeatureLink(channel='awgn').eval(),'awgn','cpu');rows=[]
            try:
                for i in range(1,7):
                    sensors,gt=helper.sensors(frame=f'{i:06d}',empty=i==6)
                    snr=controller.prepare(i);controller.link.channel(torch.ones(1,62400,2),snr)
                    rows.append(dict(step=i,epoch=1,channel='awgn',optimization_scope=scope,snr_db=snr,frame_id=f'{i:06d}',GT_shape=list(gt.shape),augmented_evidence=c.augmented_evidence(sensors,gt),channel_condition=copy.deepcopy(controller.last_evidence)))
            finally:controller.close()
            arms.append(rows)
        with tempfile.TemporaryDirectory() as directory:
            paths=[Path(directory)/f'{i}.jsonl' for i in range(2)]
            def save():
                for path,rows in zip(paths,arms):path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            save();self.assertEqual(c.paired_records(*paths,6)['state'],'passed')
            original=copy.deepcopy(arms[1][0]);arms[1][0]['channel_condition']['noise_rng_after_sha256']='0'*64;save()
            with self.assertRaises(ValueError):c.paired_records(*paths,6)
            arms[1][0]=original;arms[1][0]['augmented_evidence']['arrays']['left_img']['sha256']='0'*64;save()
            with self.assertRaises(ValueError):c.paired_records(*paths,6)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();assert not args.output.exists() and not torch.cuda.is_available()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    record=dict(state='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),torch=torch.__version__,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),source_sha256={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in HERE.glob('*.py')})
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(record,indent=2)+'\n')
    if not result.wasSuccessful():raise SystemExit(1)
if __name__=='__main__':main()
