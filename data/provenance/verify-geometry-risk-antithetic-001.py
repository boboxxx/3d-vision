"""Independent fresh8 paired-noise replay, native sensors, full states and fixed chronology."""
import argparse,hashlib,importlib.util,json,math,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];PREFIX='geometry-risk-antithetic-001'
helper_path=ROOT/'data/provenance/verify-geometry-risk-pilot-001.py'
spec=importlib.util.spec_from_file_location('sealed_full64_audit',helper_path)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
read=helper.read;sha=helper.sha;array_sha=helper.array_sha;terminal=helper.terminal
input_check=helper.input_check;geometry_check=helper.geometry_check
group_map=helper.group_map;independent_groups=helper.independent_groups;SEQUENCE=helper.SEQUENCE


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists() and not torch.cuda.is_available()
    manifest=ROOT/'data/runs'/f'{PREFIX}.json';run=read(manifest)
    launchpath=ROOT/'data/runs'/f'{PREFIX}-launch.json';launch=read(launchpath)
    assert run['state']=='finished8_paired_observations_independent_audit_pending'
    assert run['pid']==launch['pid'] and terminal(run['pid'])
    assert run['optimizer_updates']==0 and run['readonly_states']==535 and run['no_parameter_gradients']
    assert run['source_identities']==run['source_identities_after']
    roots={'project':ROOT,'liga':ROOT/'third_party/LIGA-Stereo','mmdet':ROOT/'third_party/mmdetection_kitti',
           'stereo_rcnn':ROOT/'third_party/Stereo-RCNN','F7':ROOT,'final_evaluation':ROOT,'experiment':ROOT,'pilot':ROOT,'antithetic':ROOT}
    for key,item in run['source_identities'].items():
        files={p:sha(roots[key]/p) for p in item['file_hashes']}
        assert files==item['file_hashes'] and hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()==item['sha256']
    for path,value in run['metadata_identities'].items():assert sha(ROOT/path)==value
    cp=ROOT/'data/engineering/geometry-risk-antithetic-sheng-CPU-001.json'
    assert sha(cp)==launch['CPU_gate_sha256']==run['paired_CPU_gate_sha256'] and read(cp)['tests']==6 and read(cp)['state']=='passed'
    for path,value in launch['source_sha256'].items():assert sha(ROOT/path)==value
    for path,value in run['eligibility']['artifacts_sha256'].items():assert sha(ROOT/path)==value
    old=read(ROOT/'data/runs/geometry-risk-pilot-001.json');oldaudit=read(ROOT/'data/provenance/geometry-risk-pilot-001-native-audit.json')
    assert oldaudit['state']=='passed_full64_independent_native_audit' and oldaudit['verifier_sha256']==sha(helper_path)
    assert oldaudit['actual_terminal_PID']==run['eligibility']['actual_terminal_PID']==74257 and terminal(74257)
    for suffix in ('ridge-audit','ridge-local-audit-001'):
        a=read(ROOT/'data/provenance'/('geometry-risk-pilot-001-'+suffix+'.json'))
        assert a['state']=='passed_independent_fixed_ridge_and_frame_bootstrap_audit' and a['result_sha256']==sha(ROOT/'data/analysis/geometry-risk-pilot-001-ridge-001.json')
    rng=np.random.Generator(np.random.PCG64(2804));assert rng.bit_generator.state==run['initial_PCG64_state']
    checkpoint=Path('/mnt/d/paper6/runs/stereo-native-task-seed17-002/checkpoint_epoch_1.pth')
    assert sha(checkpoint)==run['sole_parent_sha256']=='77bc73021a139099e3d0f9c12e0fba8b17eb62f9b9c243eb8f0ad099bd6e8867'
    parent=torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state']
    parent_hashes={name:array_sha(value) for name,value in parent.items()};del parent
    assert len(parent_hashes)==535 and run['all535_final_state_hashes']==parent_hashes
    directory=Path(run['output_directory']);records=directory/'native-loss.jsonl';assert sha(records)==run['records_sha256']
    rows=[json.loads(line) for line in records.read_text().splitlines()];assert len(rows)==8208
    fold=read(ROOT/'data/internal-tuning-fold-001.json');ids=sorted(fold['folds']['geocomm_tune_train']['ids'])[64:72]
    assert run['ids']==ids and len(ids)==8 and not set(ids)&set(old['fit_ids']+old['out_of_fit_ids'])
    assert not set(ids)&set(fold['folds']['geocomm_tune_holdout']['ids']) and set(run['frames'])==set(ids) and run['completed_frames']==8
    valpath=Path(run['dataset']['root'])/'ImageSets/val.txt'
    assert sha(valpath)==run['dataset']['mainval_split_sha256']==fold['main_validation_split_sha256'] and not set(ids)&set(valpath.read_text().splitlines())
    mapping=independent_groups();summaries={};total_energy=0.;draw_count=0
    artifacts={str(manifest.relative_to(ROOT)):sha(manifest),str(launchpath.relative_to(ROOT)):sha(launchpath),str(cp.relative_to(ROOT)):sha(cp)}
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
        geo=geometry_check(f);frame_rows=rows[frame_index*1026:(frame_index+1)*1026];positive_noise=None
        for index,row in enumerate(frame_rows):
            cond=row['condition'];assert row['frame_id']==cond['frame_id']==frame and row['sequence']==SEQUENCE
            assert cond['attempt']==frame_index*1026+index+1 and cond['PCG64']['before']==rng.bit_generator.state
            mode='clean' if index==0 else 'full_awgn' if index==1025 else 'group'
            group=(index-1)//32 if mode=='group' else None
            pair=((index-1)//2)%16 if mode=='group' else None
            sign=(1 if index%2 else -1) if mode=='group' else None
            assert (cond['mode'],cond['group'],cond['pair'],cond['draw'],cond['sign'])==(mode,group,pair,pair,sign)
            if mode=='clean':noise=np.zeros((62400,2),np.float32)
            elif sign==-1:noise=-positive_noise;positive_noise=None
            else:
                noise=rng.standard_normal((62400,2))*math.sqrt(.05);draw_count+=1
                if mode=='group':noise[mapping!=group]=0
                noise=noise.astype(np.float32)
                if mode=='group':positive_noise=noise.copy()
            assert cond['PCG64']['after']==rng.bit_generator.state and cond['PCG64']['independent_draw']==(mode=='full_awgn' or sign==1)
            assert cond['PCG64']['draw_shape']==([62400,2] if mode=='full_awgn' or sign==1 else [0,2])
            assert cond['PCG64']['selected_symbols']==(int((mapping==group).sum()) if mode=='group' else 62400 if mode=='full_awgn' else 0)
            assert cond['PCG64']['attempted_complex_uses']==62400 and cond['PCG64']['local_N0']==(0. if mode=='clean' else .1)
            assert cond['noise_energy']==float(np.square(noise.astype(np.float64)).sum())
            path=Path(cond['noise_path']);assert sha(path)==cond['noise_file_sha256'];actual=np.load(path,allow_pickle=False)
            assert actual.tobytes()==noise.tobytes() and array_sha(noise)==cond['noise_array_sha256']
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
                assert math.isclose(row['gradient_norms']['symbols'],float(np.linalg.norm(f['gradient'].astype(np.float64))),rel_tol=3e-6,abs_tol=1e-6)
            else:assert not row['gradient_norms'] and row['loss_minus_clean']==row['loss']-summary['clean_loss']
            if mode=='group':
                linear=float(np.dot(f['gradient'].astype(np.float64).ravel(),noise.astype(np.float64).ravel()))
                assert math.isclose(row['linear_gradient_dot_noise'],linear,rel_tol=3e-12,abs_tol=3e-12)
        assert positive_noise is None and frame_rows[-1]['loss']==summary['full_AWGN10_loss']
        for group,item in enumerate(summary['damage']):
            selected=frame_rows[group*32+1:group*32+33]
            p=np.asarray([r['loss'] for r in selected[::2]]);n=np.asarray([r['loss'] for r in selected[1::2]])
            linear=np.asarray([r['linear_gradient_dot_noise'] for r in selected[::2]])
            np.testing.assert_array_equal(-linear,[r['linear_gradient_dot_noise'] for r in selected[1::2]])
            dp=p-summary['clean_loss'];dn=n-summary['clean_loss'];q=(dp+dn)/2;s=(dp-dn)/2
            vectors=dict(positive_loss=p,negative_loss=n,positive_damage=dp,negative_damage=dn,symmetric=q,antisymmetric=s,linear=linear,antisymmetric_minus_linear=s-linear)
            assert item['group']==group
            for name,value in vectors.items():np.testing.assert_array_equal(item[name],value)
            scalars=dict(signed_symmetric_mean=float(q.mean()),positive_symmetric_mean=max(float(q.mean()),0.),symmetric_sample_variance=float(q.var(ddof=1)),
                         antisymmetric_mean=float(s.mean()),antisymmetric_sample_variance=float(s.var(ddof=1)),linear_sample_variance=float(linear.var(ddof=1)))
            for name,value in scalars.items():assert item[name]==value
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
            assert item['first_order_loss_variance_proxy']==score['first_order_loss_variance_proxy']
        summaries[frame]=dict(geometry_max_errors=geo,clean_loss=summary['clean_loss'],full_AWGN10_loss=summary['full_AWGN10_loss'],readonly_states=535,
                              frame_report_sha256=sha(frame_path),fixture_sha256=sha(fixture_path),source_files=run['frames'][frame]['source_files'])
        artifacts[str(frame_path)]=sha(frame_path)
    assert run['calls']==dict(steps=8208,student=16416,codec=8208,channel=8208,build_cost=8208,map_to_bev=8208,BEV=8208,head3D=8208,forbidden=0)
    assert run['channel_attempts']==8208 and run['attempted_complex_uses']==8208*62400 and run['final_PCG64_state']==rng.bit_generator.state
    assert draw_count==run['independent_noise_draws']==4104 and run['peak_reserved_bytes']+2*2**30<=run['physical_free_before_bytes']
    log=ROOT/'logs'/f'{PREFIX}.log'
    output=dict(state='passed_full8_independent_antithetic_native_audit',checked_unix=time.time(),verifier_sha256=sha(Path(__file__)),
                helper_sha256=sha(helper_path),evidence_type='fresh_training_scene_paired_native_loss_diagnostic_not_AP_or_allocation',
                manifest_sha256=sha(manifest),launch_sha256=sha(launchpath),native_log_sha256=sha(log),actual_terminal_PID=run['pid'],
                records_sha256=sha(records),native_rows=8208,independent_noise_draws=4104,attempted_complex_uses=8208*62400,
                actual_total_transmit_energy=total_energy,summaries=summaries,artifacts_sha256=artifacts)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(output,f,indent=2,allow_nan=False)
    print(json.dumps({'state':output['state'],'native_rows':8208,'frames':8}))


if __name__=='__main__':main()
