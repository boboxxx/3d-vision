"""Native clean fixture and sixteen paired interventions per public group."""
from probe_native import (SENSOR_INPUT_KEYS,augmented_evidence,state_hashes,epipolar_proxy,
                         aggregate_geometry,descriptor_energy,assert_frozen,sha256,save)
import numpy as np
import torch
import json
from pair_noise import pair_statistics


def measure_frame(frame,batch,model,observer,channel,directory,stream,reference):
    sensors={k:batch[k] for k in SENSOR_INPUT_KEYS}
    for k in ('left_img','right_img'):sensors[k]=torch.as_tensor(sensors[k],device='cuda',dtype=torch.float32)
    targets=torch.as_tensor(batch['gt_boxes'],device='cuda',dtype=torch.float32)
    assert tuple(sensors['left_img'].shape)==tuple(sensors['right_img'].shape)==(1,3,320,1248)
    assert str(sensors['frame_id'][0])==frame and not sensors['calib'][0].flipped and 'random_T' not in batch
    fingerprint=augmented_evidence(sensors,targets)
    fb=float(sensors['calib'][0].fu_mul_baseline);assert fb>0
    hashes_before=state_hashes(model);assert len(hashes_before)==535
    channel.begin_frame();observer.capture_clean=True;observer.clean_features=[]
    channel.prepare(frame,'clean')
    loss=observer.forward(sensors,targets);clean=float(loss.detach());symbols=channel.leaf.detach().cpu().numpy()[0].copy()
    gradient,=torch.autograd.grad(loss,channel.leaf)
    assert gradient.shape==(1,62400,2) and torch.isfinite(gradient).all()
    gradient=gradient.detach().cpu().numpy()[0].copy();row=observer.finish_backward()
    condition=channel.finish();row.update(loss=clean,condition=condition,augmented_evidence=fingerprint)
    stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
    features=(observer.clean_features[0][0],observer.clean_features[1][0],observer.clean_features[0][1])
    assert [list(f.shape) for f in features]==[[1,32,320,1248],[1,32,320,1248],[1,32,80,312]]
    observer.capture_clean=False;observer.clean_features=[];del loss
    valid_shape=(min(int(sensors['image_shape'][0][0]),320),min(int(sensors['image_shape'][0][1]),1280))
    result=epipolar_proxy(features[0][0],features[1][0],focal_baseline=fb,valid_image_shape=valid_shape)
    scores=aggregate_geometry(result,symbols,gradient,channel.group_ids)
    native_energy=descriptor_energy(features)
    for score,energy in zip(scores,native_energy):score['native_descriptor_energy_mean']=float(energy)
    arrays=dict(left_stereo=features[0],right_stereo=features[1],appearance=features[2],symbols=symbols,gradient=gradient,
                left_img=sensors['left_img'].detach().cpu().numpy(),right_img=sensors['right_img'].detach().cpu().numpy(),
                gt_boxes=targets.detach().cpu().numpy(),P2=sensors['calib'][0].P2,P3=sensors['calib'][0].P3,
                image_shape=np.asarray(sensors['image_shape']),valid_image_shape=np.asarray(valid_shape),group_ids=channel.group_ids,disparity_bins=result['disparity_bins_input_pixels'])
    for name,value in vars(sensors['calib'][0]).items():arrays['calib_'+name]=np.asarray(value)
    for view in ('left','right'):
        for name,value in result[view].items():arrays[view+'_'+name]=value
    fixture=directory/f'{frame}-clean-fixture.npz';np.savez_compressed(fixture,**arrays)
    del arrays,features,result,symbols
    changes=[];full_loss=None
    for group in range(32):
        positive=[];negative=[];linear=[]
        for pair in range(16):
            for sign in (1,-1):
                channel.prepare(frame,'group',group,pair,sign)
                linear_value=float(np.sum(gradient.astype(np.float64)*channel.noise.astype(np.float64)))
                loss=observer.forward(sensors,targets);value=float(loss.detach())
                record=observer.finish_forward();condition=channel.finish()
                assert all(p.grad is None and not p.requires_grad for p in model.parameters())
                record.update(loss=value,loss_minus_clean=value-clean,linear_gradient_dot_noise=linear_value,condition=condition,augmented_evidence_sha256=fingerprint['sha256'])
                stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();del loss
                if sign==1:positive.append(value);linear.append(linear_value)
                else:negative.append(value);assert linear_value==-linear[-1]
        statistics=pair_statistics(positive,negative,clean,linear)
        changes.append(dict(group=group,first_order_loss_variance_proxy=scores[group]['first_order_loss_variance_proxy'],**statistics))
    channel.prepare(frame,'full_awgn');loss=observer.forward(sensors,targets);full_loss=float(loss.detach())
    record=observer.finish_forward();condition=channel.finish()
    record.update(loss=full_loss,loss_minus_clean=full_loss-clean,condition=condition,augmented_evidence_sha256=fingerprint['sha256'])
    stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();del loss
    assert augmented_evidence(sensors,targets)==fingerprint
    assert assert_frozen(reference,model.state_dict(),set())==535 and state_hashes(model)==hashes_before
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    report=dict(frame_id=frame,clean_loss=clean,full_AWGN10_loss=full_loss,focal_baseline=fb,valid_image_shape=list(valid_shape),scores=scores,damage=changes,
                fixture_path=str(fixture),fixture_sha256=sha256(fixture),fixture_bytes=fixture.stat().st_size,
                augmented_evidence=fingerprint,state_hashes_before=hashes_before,state_hashes_after=state_hashes(model),
                native_loss_passes=1026,attempted_complex_uses=1026*62400,no_parameter_gradients=True)
    save(directory/f'{frame}.json',report);return report

