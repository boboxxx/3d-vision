"""Independent CPU audit of every native intervention, original inputs and geometry arrays."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[2]
PREFIX='geometry-risk-native-engineering-001'
SEQUENCE=['student','student','link_start','channel','link_done','build_cost','backbone_done','map_to_bev','BEV','GT_at_3D_head','head3D']


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def array_sha(x):
    if isinstance(x,torch.Tensor):x=x.detach().cpu().numpy()
    x=np.ascontiguousarray(x)
    return hashlib.sha256(json.dumps(dict(dtype=x.dtype.str,shape=list(x.shape)),sort_keys=True,separators=(',',':')).encode()+b'\0'+x.tobytes()).hexdigest()


def terminal(pid):
    try:os.kill(pid,0)
    except ProcessLookupError:return True
    return False


def group_map(h,w):
    return np.floor((np.arange(h)+.5)*4/h).astype(int)[:,None]*8+np.floor((np.arange(w)+.5)*8/w).astype(int)[None]


def independent_groups():
    result=[]
    for channels,h,w in [(2,20,1248),(2,20,1248),(8,20,156)]:
        result.extend(np.repeat(group_map(h,w).ravel(),channels//2))
    return np.asarray(result,dtype=np.int64)


def input_check(f,frame,data_root,evidence):
    original_shape=None
    for view,key in [('image_2','left_img'),('image_3','right_img')]:
        image=np.asarray(Image.open(data_root/'training'/view/(frame+'.png')).convert('RGB'))
        original_shape=np.asarray(image.shape[:2],np.int32)
        crop=image[max(image.shape[0]-320,0):,:1280]
        normalized=(crop.astype(np.float32)/255-np.asarray([.485,.456,.406],np.float32))/np.asarray([.229,.224,.225],np.float32)
        normalized=np.pad(normalized,((0,(32-normalized.shape[0]%32)%32),(0,(32-normalized.shape[1]%32)%32),(0,0)))
        np.testing.assert_array_equal(f[key],normalized.transpose(2,0,1)[None])
    np.testing.assert_array_equal(f['image_shape'],original_shape[None])
    expected_boxes=[]
    classes={'Car':1,'Van':1,'Pedestrian':2,'Person_sitting':2,'Cyclist':3}
    for line in (data_root/'training/label_2'/(frame+'.txt')).read_text().splitlines():
        words=line.split()
        if not words or words[0] not in classes:continue
        h,w,l,x,y,z,angle=np.asarray([float(v) for v in words[8:15]],np.float32)
        # Native pseudo-LiDAR convention, bottom-centre camera annotation to box centre.
        expected_boxes.append([z,-x,-float(y)+float(np.float32(h/2)),l,w,h,-float(np.float32(angle+np.float32(np.pi/2))),classes[words[0]]])
    expected_boxes=np.asarray(expected_boxes,np.float32).reshape(1,-1,8)
    # Deterministic TEST augmentor also wraps yaw, independent explicit expression.
    period=np.float32(2*np.pi)
    expected_boxes[:,:,6]-=np.floor(expected_boxes[:,:,6]/period+np.float32(.5))*period
    np.testing.assert_allclose(f['gt_boxes'],expected_boxes,rtol=2e-6,atol=2e-6)
    calibration={}
    for line in (data_root/'training/calib'/(frame+'.txt')).read_text().splitlines():
        if ':' in line:
            key,value=line.split(':',1);calibration[key]=np.fromstring(value,sep=' ',dtype=np.float32)
    offset=int(original_shape[0])-min(int(original_shape[0]),320)
    for name in ('P2','P3'):
        matrix=calibration[name].reshape(3,4).copy();matrix[1]-=offset*matrix[2]
        # The native FP32 K @ inv(K) decomposition has documented rounding.
        np.testing.assert_allclose(f['calib_'+name],matrix,atol=3e-4,rtol=3e-6)
    np.testing.assert_array_equal(f['calib_R0'],calibration['R0_rect'].reshape(3,3))
    np.testing.assert_array_equal(f['calib_V2C'],calibration['Tr_velo_to_cam'].reshape(3,4))
    np.testing.assert_array_equal(f['calib_offsets'],[0,offset]);assert not f['calib_flipped'].item()
    for name in ('left_img','right_img','image_shape','gt_boxes'):
        assert array_sha(f[name])==evidence['arrays'][name]['sha256']
    for name,value in evidence['calibration'].items():assert array_sha(f['calib_'+name])==value['sha256']


def geometry_check(f):
    descriptors=[];valid=[]
    height,width=map(int,f['valid_image_shape']);max_h,max_w=height//4,width//4
    for name in ('left_stereo','right_stereo'):
        original=f[name][0];c,h,w=original.shape
        pooled=original.astype(np.float64).reshape(c,h//4,4,w//4,4).mean(axis=(2,4))
        norm=np.linalg.norm(pooled,axis=0);unit=np.divide(pooled,norm[None],out=np.zeros_like(pooled),where=norm[None]>np.finfo(original.dtype).eps)
        angular=np.zeros_like(norm)
        for axis in (1,2):
            diff=np.sum(np.diff(unit,axis=axis)**2,axis=0)
            if axis==1:angular[1:]+=diff;angular[:-1]+=diff
            else:angular[:,1:]+=diff;angular[:,:-1]+=diff
        support=(norm>np.finfo(original.dtype).eps)&(angular>1e-12)
        support[np.arange(support.shape[0])>=max_h]=False;support[:,np.arange(support.shape[1])>=max_w]=False
        descriptors.append(unit);valid.append(support)
    _,h,w=descriptors[0].shape;fb=float(abs(f['calib_P2'][0,3]-f['calib_P3'][0,3]));depth=fb/(4*np.arange(1,49,dtype=np.float64))
    max_errors={}
    for view in ('left','right'):
        cosine=np.zeros((48,h,w));support=np.zeros((48,h,w),bool)
        for disparity in range(1,49):
            correlations=np.sum(descriptors[0][:,:,disparity:]*descriptors[1][:,:,:-disparity],axis=0)
            possible=valid[0][:,disparity:]&valid[1][:,:-disparity]
            target=slice(disparity,None) if view=='left' else slice(None,-disparity)
            cosine[disparity-1,:,target]=correlations;support[disparity-1,:,target]=possible
        any_support=support.any(axis=0)
        scaled=np.where(support,cosine/.1,-np.inf)
        maximum=np.max(scaled,axis=0);maximum=np.where(any_support,maximum,0)
        unnormalized=np.exp(scaled-maximum[None]);denominator=unnormalized.sum(axis=0)
        probability=np.divide(unnormalized,denominator[None],out=np.zeros_like(unnormalized),where=denominator[None]>0)
        np.testing.assert_array_equal(f[view+'_valid'],any_support)
        np.testing.assert_array_equal(f[view+'_support_count'],support.sum(axis=0))
        np.testing.assert_allclose(f[view+'_probability'],probability,rtol=2e-12,atol=2e-12)
        d=4*np.arange(1,49,dtype=np.float64)[:,None,None];z=depth[:,None,None]
        dm=(probability*d).sum(axis=0);zm=(probability*z).sum(axis=0)
        lp=np.log(probability,out=np.zeros_like(probability),where=probability>0)
        values=dict(entropy=-(probability*lp).sum(axis=0),depth_mean=zm,depth_variance=(probability*(z-zm[None])**2).sum(axis=0),
                    disparity_mean=dm,disparity_variance=(probability*(d-dm[None])**2).sum(axis=0))
        for name,value in values.items():
            expected=np.where(any_support,value,np.nan);actual=f[view+'_'+name]
            np.testing.assert_allclose(actual,expected,rtol=3e-12,atol=3e-12,equal_nan=True)
            max_errors[view+'_'+name]=float(np.max(np.abs(actual[any_support]-expected[any_support]))) if any_support.any() else None
    return max_errors


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists() and not torch.cuda.is_available(), 'unique CPU-only audit required'
    manifest=ROOT/'data/engineering'/f'{PREFIX}.json';run=read(manifest)
    queue=read(ROOT/'data/runs'/f'{PREFIX}-queue.json')
    assert run['state']=='finished_native_engineering_audit_pending' and queue['state']=='native_engineering_terminal_independent_audit_pending'
    assert terminal(run['pid']) and terminal(queue['pid'])
    assert queue['native_manifest_sha256']==sha(manifest) and queue['native_log_sha256']==sha(ROOT/'logs'/f'{PREFIX}-native.log')
    assert run['optimizer_updates']==0 and run['readonly_states']==535 and run['no_parameter_gradients']
    assert run['source_identities']==run['source_identities_after']==queue['source_identities']
    roots={'project':ROOT,'liga':ROOT/'third_party/LIGA-Stereo','mmdet':ROOT/'third_party/mmdetection_kitti',
           'stereo_rcnn':ROOT/'third_party/Stereo-RCNN','F7':ROOT,'final_evaluation':ROOT,'experiment':ROOT}
    for key,item in run['source_identities'].items():
        files={name:sha(roots[key]/name) for name in item['file_hashes']}
        assert files==item['file_hashes'] and hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()==item['sha256']
    for name,value in run['metadata_identities'].items():assert sha(ROOT/name)==value
    cpu_path=ROOT/'data/engineering/geometry-risk-native-sheng-CPU-001.json';cpu=read(cpu_path)
    assert cpu['state']=='passed' and cpu['tests']==8 and sha(cpu_path)==run['CPU_gate_sha256']==queue['CPU_gate_sha256']
    checkpoint=Path('/mnt/d/paper6/runs/stereo-native-task-seed17-002/checkpoint_epoch_1.pth')
    assert sha(checkpoint)==run['sole_parent_sha256']=='77bc73021a139099e3d0f9c12e0fba8b17eb62f9b9c243eb8f0ad099bd6e8867'
    parent=torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state'];parent_hashes={name:array_sha(value) for name,value in parent.items()}
    assert len(parent_hashes)==535 and run['all535_final_state_hashes']==parent_hashes;del parent
    directory=Path(run['output_directory']);records=directory/'native-loss.jsonl';assert sha(records)==run['records_sha256']
    rows=[json.loads(line) for line in records.read_text().splitlines()];assert len(rows)==260
    ids=read(ROOT/'experiments/geometry-risk/native-engineering-inputs-001.json')['engineering_ids'];assert ids==['000000','000003']
    mapping=independent_groups();rng=np.random.Generator(np.random.PCG64(2801));artifacts={str(manifest.relative_to(ROOT)):sha(manifest),str(cpu_path.relative_to(ROOT)):sha(cpu_path)}
    summaries={};total_energy=0.
    for frame_index,frame in enumerate(ids):
        frame_path=Path(run['frames'][frame]['report_path']);summary=read(frame_path);assert sha(frame_path)==run['frames'][frame]['report_sha256']
        assert summary['state_hashes_before']==summary['state_hashes_after']==parent_hashes and summary['no_parameter_gradients']
        for name,value in run['frames'][frame]['source_files'].items():assert sha(Path(name))==value
        fixture_path=Path(summary['fixture_path']);assert sha(fixture_path)==summary['fixture_sha256'] and fixture_path.stat().st_size==summary['fixture_bytes']
        with np.load(fixture_path,allow_pickle=False) as source:f={k:source[k] for k in source.files}
        data_root=Path(run['dataset']['root']);input_check(f,frame,data_root,summary['augmented_evidence'])
        np.testing.assert_array_equal(f['group_ids'],mapping);assert f['symbols'].shape==f['gradient'].shape==(62400,2)
        assert np.isfinite(f['symbols']).all() and np.isfinite(f['gradient']).all()
        assert abs(np.square(f['symbols'].astype(np.float64)).sum()/62400-1)<1e-4
        assert summary['focal_baseline']==float(abs(f['calib_P2'][0,3]-f['calib_P3'][0,3]))
        assert summary['valid_image_shape']==f['valid_image_shape'].tolist()
        geo=geometry_check(f);frame_rows=rows[frame_index*130:(frame_index+1)*130]
        for index,row in enumerate(frame_rows):
            cond=row['condition'];assert row['frame_id']==cond['frame_id']==frame and row['sequence']==SEQUENCE
            assert cond['attempt']==frame_index*130+index+1 and cond['PCG64']['before']==rng.bit_generator.state
            mode='clean' if index==0 else 'full_awgn' if index==129 else 'group'
            group=(index-1)//4 if mode=='group' else None;draw=(index-1)%4 if mode=='group' else None
            assert (cond['mode'],cond['group'],cond['draw'])==(mode,group,draw)
            if mode=='clean':noise=np.zeros((62400,2),np.float32)
            else:
                noise=rng.standard_normal((62400,2))*math.sqrt(.1/2)
                if mode=='group':noise[mapping!=group]=0
                noise=noise.astype(np.float32)
            assert cond['PCG64']['after']==rng.bit_generator.state
            path=Path(cond['noise_path']);assert sha(path)==cond['noise_file_sha256'];actual=np.load(path,allow_pickle=False)
            np.testing.assert_array_equal(actual,noise);assert array_sha(noise)==cond['noise_array_sha256']
            expected=f['symbols'] if mode=='clean' else f['symbols']+noise
            assert array_sha(f['symbols'][None])==cond['transmitted_symbols_sha256'] and array_sha(expected[None])==cond['received_symbols_sha256']
            assert cond['global_rng_before']==cond['global_rng_after'] and cond['consumed']
            assert row['total_complex_uses']==62400 and row['pilot_complex_uses']==row['header_complex_uses']==0
            assert row['GT_introduced_only_at_head'] and row['cost_inputs_are_received_features'] and row['appearance_input_is_received_feature'] and row['all_modules_eval']
            expected_loss=row['classification_loss']+row['location_loss']+row['direction_loss']+row['iou_loss_raw']*row['iou_weight']
            assert math.isclose(row['loss'],expected_loss,rel_tol=3e-6,abs_tol=1e-6) and math.isfinite(row['loss'])
            assert abs(row['tx_energy']/62400-1)<1e-4;total_energy+=row['tx_energy']
            if mode=='clean':
                assert row['loss']==summary['clean_loss'] and set(row['gradient_norms'])=={'symbols','native_cost','left_stereo','right_stereo','appearance'}
                assert math.isclose(row['gradient_norms']['symbols'],float(np.linalg.norm(f['gradient'])),rel_tol=3e-6,abs_tol=1e-6)
            else:assert not row['gradient_norms'] and row['loss_minus_clean']==row['loss']-summary['clean_loss']
        assert frame_rows[-1]['loss']==summary['full_AWGN10_loss']
        for group,item in enumerate(summary['damage']):
            deltas=[r['loss']-summary['clean_loss'] for r in frame_rows[group*4+1:group*4+5]]
            assert item['group']==group and item['draw_damage']==deltas and item['signed_damage_mean']==float(np.mean(deltas))
            assert item['positive_damage_mean']==max(float(np.mean(deltas)),0.) and item['sample_variance']==float(np.var(deltas,ddof=1))
            score=summary['scores'][group];mask=mapping==group
            assert score['group']==group and score['symbol_count']==int(mask.sum())
            assert math.isclose(score['code_energy_mean'],float(np.square(f['symbols'][mask].astype(np.float64)).sum(1).mean()),rel_tol=1e-12)
            assert score['squared_gradient_sum']==float(np.square(f['gradient'][mask].astype(np.float64)).sum())
            assert score['first_order_loss_variance_proxy']==.05*score['squared_gradient_sum']
            cell_ids=group_map(*f['left_valid'].shape)
            valid_masks=[(cell_ids==group)&f[view+'_valid'] for view in ('left','right')]
            assert score['valid_count']==sum(int(m.sum()) for m in valid_masks)
            assert score['candidate_count']==2*int((cell_ids==group).sum())
            for field in ('entropy','depth_mean','depth_variance','disparity_mean','disparity_variance','support_count'):
                values=np.concatenate([f[view+'_'+field][m] for view,m in zip(('left','right'),valid_masks)])
                if values.size:assert math.isclose(score[field],float(values.mean()),rel_tol=3e-12,abs_tol=3e-12)
                else:assert score[field] is None
            native_energies=[]
            for field in ('left_stereo','right_stereo','appearance'):
                energy=np.square(f[field][0].astype(np.float64)).mean(axis=0);cells=group_map(*energy.shape)
                native_energies.append(float(energy[cells==group].mean()))
            assert math.isclose(score['native_descriptor_energy_mean'],float(np.mean(native_energies)),rel_tol=3e-12,abs_tol=3e-12)
        summaries[frame]=dict(geometry_max_errors=geo,clean_loss=summary['clean_loss'],full_AWGN10_loss=summary['full_AWGN10_loss'],readonly_states=535,
                              frame_report_sha256=sha(frame_path),fixture_sha256=sha(fixture_path),source_files=run['frames'][frame]['source_files'])
        artifacts[str(frame_path)]=sha(frame_path)
    assert run['calls']==dict(steps=260,student=520,codec=260,channel=260,build_cost=260,map_to_bev=260,BEV=260,head3D=260,forbidden=0)
    assert run['channel_attempts']==260 and run['attempted_complex_uses']==260*62400 and run['final_PCG64_state']==rng.bit_generator.state
    output=dict(state='passed_independent_native_engineering_audit',checked_unix=time.time(),verifier_sha256=sha(Path(__file__)),
                evidence_type='engineering_only_no_fit_AP_allocation_or_calibration_claim',manifest_sha256=sha(manifest),queue_manifest_sha256=sha(ROOT/'data/runs'/f'{PREFIX}-queue.json'),
                actual_terminal_PIDs=[run['pid'],queue['pid']],records_sha256=sha(records),native_rows=260,attempted_complex_uses=260*62400,
                actual_total_transmit_energy=total_energy,summaries=summaries,artifacts_sha256=artifacts)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'state':output['state'],'native_rows':260,'attempted_complex_uses':260*62400}))


if __name__=='__main__':main()
