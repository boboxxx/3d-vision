"""Full64 independent CPU audit; frozen raw-input/geometry helpers from engineering."""
import argparse,hashlib,importlib.util,json,math,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];PREFIX='geometry-risk-pilot-001'
helper_path=ROOT/'data/provenance/verify-geometry-risk-native-engineering-001.py'
spec=importlib.util.spec_from_file_location('sealed_native_audit',helper_path)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
read=helper.read;sha=helper.sha;array_sha=helper.array_sha;terminal=helper.terminal
geometry_check=helper.geometry_check
group_map=helper.group_map;independent_groups=helper.independent_groups;SEQUENCE=helper.SEQUENCE


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
    # Native TEST mode does not remap Van or Person_sitting.
    classes={'Car':1,'Pedestrian':2,'Cyclist':3}
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



def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists() and not torch.cuda.is_available(), 'unique CPU-only full observation audit'
    manifest=ROOT/'data/runs'/f'{PREFIX}.json';run=read(manifest)
    launchpath=ROOT/'data/runs'/f'{PREFIX}-launch.json';launch=read(launchpath)
    assert run['state']=='finished64_native_observations_independent_audit_pending'
    assert run['pid']==launch['pid'] and terminal(run['pid'])
    assert run['optimizer_updates']==0 and run['readonly_states']==535 and run['no_parameter_gradients']
    assert run['source_identities']==run['source_identities_after']
    roots={'project':ROOT,'liga':ROOT/'third_party/LIGA-Stereo','mmdet':ROOT/'third_party/mmdetection_kitti',
           'stereo_rcnn':ROOT/'third_party/Stereo-RCNN','F7':ROOT,'final_evaluation':ROOT,'experiment':ROOT,'pilot':ROOT}
    for key,item in run['source_identities'].items():
        files={p:sha(roots[key]/p) for p in item['file_hashes']}
        assert files==item['file_hashes'] and hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()==item['sha256']
    for path,value in run['metadata_identities'].items():assert sha(ROOT/path)==value
    cpu_path=ROOT/'data/engineering/geometry-risk-native-sheng-CPU-001.json'
    assert sha(cpu_path)==run['CPU_gate_sha256'] and read(cpu_path)['tests']==8
    stats_cpu=ROOT/'data/engineering/geometry-risk-pilot-sheng-CPU-001.json'
    assert sha(stats_cpu)==launch['CPU_gate_sha256'] and read(stats_cpu)['state']=='passed' and read(stats_cpu)['tests']==5
    for path,value in launch['source_sha256'].items():assert sha(ROOT/path)==value
    engineering_path=ROOT/'data/engineering/geometry-risk-native-engineering-001.json'
    engineering_audit_path=ROOT/'data/provenance/geometry-risk-native-engineering-001-audit.json'
    eng=read(engineering_path);engaudit=read(engineering_audit_path)
    assert sha(engineering_path)==run['eligibility']['manifest_sha256']==engaudit['manifest_sha256']
    assert sha(engineering_audit_path)==run['eligibility']['audit_sha256'] and sha(helper_path)==engaudit['verifier_sha256']
    assert engaudit['state']=='passed_independent_native_engineering_audit'
    rng=np.random.Generator(np.random.PCG64(2801))
    for _ in range(258):rng.standard_normal((62400,2))
    assert rng.bit_generator.state==eng['final_PCG64_state']==run['initial_PCG64_state']==run['eligibility']['continued_PCG64_state']
    checkpoint=Path('/mnt/d/paper6/runs/stereo-native-task-seed17-002/checkpoint_epoch_1.pth')
    assert sha(checkpoint)==run['sole_parent_sha256']=='77bc73021a139099e3d0f9c12e0fba8b17eb62f9b9c243eb8f0ad099bd6e8867'
    parent=torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state']
    parent_hashes={name:array_sha(value) for name,value in parent.items()};del parent
    assert len(parent_hashes)==535 and run['all535_final_state_hashes']==parent_hashes
    directory=Path(run['output_directory']);records=directory/'native-loss.jsonl';assert sha(records)==run['records_sha256']
    rows=[json.loads(line) for line in records.read_text().splitlines()];assert len(rows)==8320
    lock=read(ROOT/'experiments/geometry-risk/native-engineering-inputs-001.json')
    ids=lock['future_fit_ids']+lock['future_out_of_fit_ids']
    assert run['fit_ids']==lock['future_fit_ids'] and run['out_of_fit_ids']==lock['future_out_of_fit_ids']
    assert len(ids)==len(set(ids))==64 and set(run['frames'])==set(ids) and run['completed_frames']==64
    mapping=independent_groups();summaries={};total_energy=0.
    artifacts={str(manifest.relative_to(ROOT)):sha(manifest),str(launchpath.relative_to(ROOT)):sha(launchpath),str(stats_cpu.relative_to(ROOT)):sha(stats_cpu)}
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
                # Independent reference accumulates all 124800 components in FP64.
                assert math.isclose(row['gradient_norms']['symbols'],float(np.linalg.norm(f['gradient'].astype(np.float64))),rel_tol=3e-6,abs_tol=1e-6)
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
    assert run['calls']==dict(steps=8320,student=16640,codec=8320,channel=8320,build_cost=8320,map_to_bev=8320,BEV=8320,head3D=8320,forbidden=0)
    assert run['channel_attempts']==8320 and run['attempted_complex_uses']==8320*62400 and run['final_PCG64_state']==rng.bit_generator.state
    assert run['peak_reserved_bytes']+2*2**30<=run['physical_free_before_bytes']
    log=ROOT/'logs'/f'{PREFIX}.log'
    output=dict(state='passed_full64_independent_native_audit',checked_unix=time.time(),verifier_sha256=sha(Path(__file__)),
                helper_sha256=sha(helper_path),evidence_type='training_only_native_loss_diagnostic_not_mainval_AP_allocation',
                manifest_sha256=sha(manifest),launch_sha256=sha(launchpath),native_log_sha256=sha(log),actual_terminal_PID=run['pid'],
                records_sha256=sha(records),native_rows=8320,attempted_complex_uses=8320*62400,
                actual_total_transmit_energy=total_energy,summaries=summaries,artifacts_sha256=artifacts)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(output,f,indent=2,allow_nan=False)
    print(json.dumps({'state':output['state'],'native_rows':8320,'frames':64}))


if __name__=='__main__':main()
