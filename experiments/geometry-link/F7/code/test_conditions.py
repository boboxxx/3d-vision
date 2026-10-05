import copy,importlib.util,json,random,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('F7_conditions',HERE/'conditions.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
from geocomm.stereo_feature_link import StereoFeatureLink

def sensors(frame='001000',empty=False,flipped=False):
    calib=SimpleNamespace(P2=np.arange(12,dtype=np.float64).reshape(3,4),
      P3=np.arange(12,dtype=np.float64).reshape(3,4)+1,offsets=[0,55],flipped=flipped)
    if not flipped:calib.R0=np.eye(3);calib.V2C=np.arange(12,dtype=np.float64).reshape(3,4)
    batch=dict(batch_size=1,frame_id=[frame],left_img=np.ones((1,3,16,32),dtype=np.float32),
      right_img=np.full((1,3,16,32),2.,dtype=np.float32),calib=[calib],image_shape=np.array([[375,1242]],dtype=np.int32),random_T=np.eye(4)[None])
    return batch,np.empty((1,0,8),dtype=np.float32) if empty else np.ones((1,2,8),dtype=np.float32)

class Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_CUDA_alias_is_resolved_to_current_index_without_accepting_other_devices(self):
        with patch('torch.cuda.current_device',return_value=2) as current:
            self.assertEqual(c.generator_device('cuda'),torch.device('cuda:2'))
            self.assertNotEqual(c.generator_device('cuda'),torch.device('cuda:0'))
            self.assertEqual(current.call_count,2)
            self.assertEqual(c.generator_device('cuda:0'),torch.device('cuda:0'))
            self.assertEqual(c.generator_device('cpu'),torch.device('cpu'))
            self.assertEqual(current.call_count,2)

    def test_SNR_schedule_does_not_consume_global_RNG_and_rejects_different_seed(self):
        py=random.getstate();npstate=np.random.get_state();torchstate=torch.get_rng_state().clone()
        a=c.schedule_manifest();b=c.schedule_manifest()
        self.assertEqual(a,b);self.assertEqual(len(a['SNR_values']),3340)
        self.assertTrue(all(0<=v<=20 for v in a['SNR_values']))
        self.assertEqual(random.getstate(),py);self.assertTrue(np.array_equal(np.random.get_state()[1],npstate[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(),torchstate))
        other=random.Random(1708);self.assertNotEqual(a['SNR_values'][:6],[other.uniform(0,20) for _ in range(6)])

    def test_AWGN_is_dedicated_exact_replay_with_native_hooks_grad_and_unchanged_states(self):
        link=StereoFeatureLink(channel='awgn').eval();before={k:v.clone() for k,v in link.state_dict().items()}
        controller=c.PairedChannel(link,'awgn','cpu');symbols=torch.ones(1,62400,2,requires_grad=True)
        global_state=torch.get_rng_state().clone();hooks=[]
        handle=link.channel.register_forward_pre_hook(lambda m,a:hooks.append(a[0]))
        try:
            snr=controller.prepare(1);out,account=link.channel(symbols,snr)
            replay=torch.Generator().manual_seed(1708)
            expected=symbols+torch.randn(symbols.shape,generator=replay)*(10**(-snr/10)/2)**.5
            self.assertTrue(torch.equal(out,expected));self.assertIs(hooks[0],symbols)
            self.assertTrue(torch.equal(torch.get_rng_state(),global_state));self.assertEqual(account['total_complex_uses'],62400)
            self.assertEqual(account['pilot_complex_uses'],0);out.square().mean().backward()
            self.assertTrue(torch.isfinite(symbols.grad).all());self.assertGreater(float(symbols.grad.norm()),0)
            self.assertTrue(all(torch.equal(v,link.state_dict()[k]) for k,v in before.items()))
            evidence=controller.checkpoint_rng();self.assertTrue(torch.equal(evidence['final_state'],replay.get_state()))
            self.assertNotEqual(evidence['initial_sha256'],evidence['final_sha256'])
            with self.assertRaises(ValueError):link.channel(symbols,snr)
        finally:handle.remove();controller.close()
        self.assertEqual(link.snr_db,10.)

    def test_identity_does_not_draw_noise_and_bad_order_generator_layout_rejected(self):
        link=StereoFeatureLink(channel='identity').eval();controller=c.PairedChannel(link,'identity','cpu');symbols=torch.ones(1,62400,2)
        try:
            with self.assertRaises(ValueError):controller.prepare(2)
            snr=controller.prepare(1)
            with self.assertRaises(ValueError):controller.prepare(1)
            with self.assertRaises(ValueError):link.channel(symbols,snr+1)
            with self.assertRaises(ValueError):link.channel(symbols,snr,generator=torch.Generator())
            with self.assertRaises(ValueError):link.channel(symbols[:,:3],snr)
            out,_=link.channel(symbols,snr);self.assertIs(out,symbols)
            e=controller.checkpoint_rng();self.assertEqual(e['initial_sha256'],e['final_sha256'])
            with self.assertRaises(ValueError):controller.prepare(1)
        finally:controller.close()
        with self.assertRaises(ValueError):controller.prepare(2)

    def test_augmented_fingerprint_captures_pixels_target_camera_transform_and_flipped_empty_GT(self):
        batch,gt=sensors();original=copy.deepcopy(batch);e=c.augmented_evidence(batch,gt);c.verify_augmented_evidence(e)
        self.assertEqual(set(batch),set(original));self.assertNotIn('augmented_evidence',batch)
        for key in ('left_img','right_img','image_shape','random_T'):
            altered=copy.deepcopy(batch);altered[key].flat[0]+=1
            self.assertNotEqual(c.augmented_evidence(altered,gt)['sha256'],e['sha256'])
        altered=copy.deepcopy(batch);altered['calib'][0].P2.flat[0]+=1
        self.assertNotEqual(c.augmented_evidence(altered,gt)['sha256'],e['sha256'])
        target=gt.copy();target.flat[0]+=1;target_e=c.augmented_evidence(batch,target)
        self.assertNotEqual(target_e['sha256'],e['sha256']);self.assertEqual(target_e['arrays']['left_img'],e['arrays']['left_img'])
        batch,gt=sensors(empty=True,flipped=True);f=c.augmented_evidence(batch,gt);c.verify_augmented_evidence(f)
        self.assertEqual(f['arrays']['gt_boxes']['shape'],[1,0,8]);self.assertNotIn('R0',f['calibration'])
        batch['gt_boxes']=gt
        with self.assertRaises(ValueError):c.augmented_evidence(batch,gt)

    def test_independent_pair_audit_rejects_mismatched_images_targets_SNR_and_noise_rng(self):
        rows={arm:[] for arm in ('identity','awgn')}
        for arm in rows:
            link=StereoFeatureLink(channel=arm).eval();controller=c.PairedChannel(link,arm,'cpu')
            try:
                for step in range(1,7):
                    batch,gt=sensors(frame=f'{1000+step:06d}',empty=step==6,flipped=step%2==0)
                    snr=controller.prepare(step);link.channel(torch.ones(1,62400,2),snr)
                    rows[arm].append(dict(step=step,epoch=1,frame_id=batch['frame_id'][0],GT_shape=list(gt.shape),channel=arm,
                       snr_db=snr,channel_condition=copy.deepcopy(controller.last_evidence),augmented_evidence=c.augmented_evidence(batch,gt)))
            finally:controller.close()
        with tempfile.TemporaryDirectory() as folder:
            paths=[Path(folder)/f'{arm}.jsonl' for arm in rows]
            def write(values):
                for path,arm in zip(paths,rows):path.write_text(''.join(json.dumps(v)+'\n' for v in values[arm]))
            write(rows);self.assertEqual(c.paired_records(*paths,6)['state'],'passed')
            for kind in ('pixels','target','snr','rng','count'):
                changed=copy.deepcopy(rows)
                if kind in ('pixels','target'):
                    batch,gt=sensors(frame='001003',flipped=False)
                    if kind=='pixels':batch['right_img'].flat[0]+=1
                    else:gt.flat[0]+=1
                    changed['awgn'][2]['augmented_evidence']=c.augmented_evidence(batch,gt)
                elif kind=='snr':changed['awgn'][2]['snr_db']+=.1
                elif kind=='rng':changed['awgn'][2]['channel_condition']['noise_rng_before_sha256']='f'*64
                else:changed['awgn'].pop()
                write(changed)
                with self.assertRaises(ValueError):c.paired_records(*paths,6)

if __name__=='__main__':unittest.main()
