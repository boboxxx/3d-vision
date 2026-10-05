#!/usr/bin/env python3
"""Audit full KITTI predictions and independently repeat unchanged evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'third_party/LIGA-Stereo')]
from geocomm.evidence import sha256, serializable


def audit_prediction(anno, path):
    lines = path.read_text().splitlines()
    assert len(lines) == len(anno['name'])
    for key in ['alpha','bbox','dimensions','location','rotation_y','score']:
        assert np.isfinite(anno[key]).all(), (path,key)
    assert (anno['dimensions']>0).all()
    assert ((anno['score']>=0)&(anno['score']<=1)).all()
    for index,line in enumerate(lines):
        parts = line.split()
        assert len(parts)==16 and parts[0]==anno['name'][index]
        expected = np.r_[[-1,-1,anno['alpha'][index]],anno['bbox'][index],
            anno['dimensions'][index][[1,2,0]],anno['location'][index],
            anno['rotation_y'][index],anno['score'][index]]
        actual = np.asarray([float(x) for x in parts[1:]])
        assert np.isfinite(actual).all()
        # Writer rounds alpha/bbox to four decimals, remaining values finer.
        tolerance = np.r_[np.full(7,5.1e-5),np.full(7,5.1e-7),5.1e-9]
        assert (np.abs(actual-expected)<=tolerance).all(),path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--source-manifest',type=Path,required=True)
    parser.add_argument('--preparation',type=Path,required=True)
    parser.add_argument('--tuning-fold',type=Path,
                        help='audit codec-training auto-evaluation on its training-only holdout')
    parser.add_argument('--diagnostic-holdout',action='store_true',
                        help='uncompressed detector/encoder diagnostic, not channel comparison')
    parser.add_argument('--identity-codec-diagnostic',action='store_true',
                        help='codec present with exactly zero channel noise; not a wireless AP result')
    parser.add_argument('--stereo-feature-link',action='store_true',
                        help='F5 stereo feature boundary; exact public62400 layout')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('preserve earlier audits; choose unique output')
    run = json.loads(args.manifest.read_text())
    assert run['state']=='finished'
    fold = json.loads(args.tuning_fold.read_text()) if args.tuning_fold else None
    if args.stereo_feature_link:
        assert fold is not None and run['mode']=='test' and not args.diagnostic_holdout
        import yaml
        cfg_path=Path(run['arguments'][run['arguments'].index('--cfg_file')+1])
        assert sha256(cfg_path)==run['config_sha256']
        backbone=yaml.safe_load(cfg_path.read_text())['MODEL']['BACKBONE_3D']
        link=backbone['STEREO_FEATURE_LINK']
        assert link['enabled'] and link['snr_db']==10. and link['pilots']==8
        assert not backbone['SEMANTIC_LINK']['enabled'] and backbone['STUDENT_ENCODER']['enabled']
        assert link['channel']==('identity' if args.identity_codec_diagnostic else 'awgn')
    if args.identity_codec_diagnostic and not args.stereo_feature_link:
        assert fold is not None and run['mode']=='test' and not args.diagnostic_holdout
        import yaml
        cfg_path=Path(run['arguments'][run['arguments'].index('--cfg_file')+1])
        assert sha256(cfg_path)==run['config_sha256']
        link=yaml.safe_load(cfg_path.read_text())['MODEL']['BACKBONE_3D']['SEMANTIC_LINK']
        assert link['enabled'] and link['channel']=='identity' and link['allocation']=='uniform'
    if args.diagnostic_holdout:
        assert fold is not None and run['mode']=='test'
        dataset=run['dataset']
        assert dataset['split']=='geocomm_tune_holdout'
    elif fold and run['mode']=='train':
        assert run['codec_only'] and len(run['evaluations'])==1
        dataset=run['evaluations'][0]['dataset']
        assert dataset['split']=='geocomm_tune_holdout'
        assert run['evaluations'][0]['metrics']==run['metrics']
    else:
        assert run['mode']=='test'
        dataset=run['dataset']
    assert set(run['test_inference_inputs'])<=set(['batch_size','left_img','right_img','calib','image_shape','frame_id'])
    source = json.loads(args.source_manifest.read_text())
    for tree,key in [('project','project_sources'),('liga','detector_sources'),('mmdet','mmdet_sources')]:
        assert run[key]['sha256']==source[tree]['actual_sources']['sha256'],tree
    root = Path(dataset['root'])
    split = root/'ImageSets'/(dataset['split']+'.txt')
    ids = split.read_text().split()
    expected_count=372 if fold else 3769
    assert len(ids)==len(set(ids))==dataset['count']==expected_count
    assert sha256(split)==dataset['split_sha256']
    assert sha256(root/'download-manifest.json')==dataset['download_manifest_sha256']
    preparation = json.loads(args.preparation.read_text())
    assert preparation['state']=='finished'
    if fold:
        assert fold['state']=='finished' and not fold['main_validation_used_for_tuning']
        assert sha256(root/'ImageSets/train.txt')==fold['source_train_split_sha256']
        assert sha256(root/'ImageSets/val.txt')==fold['main_validation_split_sha256']
        assert sha256(root/'kitti_infos_train.pkl')==fold['source_train_infos_sha256']==preparation['infos']['train']['sha256']
        assert ids==fold['folds']['geocomm_tune_holdout']['ids']
        assert not set(ids)&set((root/'ImageSets/val.txt').read_text().split())
        assert not set(ids)&set(fold['folds']['geocomm_tune_train']['ids'])
        identities=[('geocomm_tune_holdout',dataset)]
        if run['mode']=='train':
            identities.insert(0,('geocomm_tune_train',run['dataset']))
        for name,identity in identities:
            assert identity['root']==str(root) and identity['split']==name
            assert identity['count']==fold['folds'][name]['count']
            assert sha256(root/'ImageSets'/(name+'.txt'))==identity['split_sha256']==fold['folds'][name]['split_sha256']
            assert (root/'ImageSets'/(name+'.txt')).read_text().split()==fold['folds'][name]['ids']
            assert sha256(root/('kitti_infos_'+name+'.pkl'))==identity['infos']['kitti_infos_'+name+'.pkl']==fold['folds'][name]['infos_sha256']
        info_path=root/'kitti_infos_geocomm_tune_holdout.pkl'
    else:
        assert dataset['split']=='val'
        info_path = Path(preparation['infos']['val']['path'])
        assert sha256(info_path)==preparation['infos']['val']['sha256']==dataset['infos']['kitti_infos_val.pkl']
    with info_path.open('rb') as stream:
        infos=pickle.load(stream)
    assert [x['point_cloud']['lidar_idx'] for x in infos]==ids
    metrics_path = Path(run['metrics']['path'])
    assert sha256(metrics_path)==run['metrics']['sha256']
    metrics = json.loads(metrics_path.read_text())
    result_dir=metrics_path.parent
    with (result_dir/'result.pkl').open('rb') as stream:
        annos=pickle.load(stream)
    assert [x['frame_id'] for x in annos]==ids
    prediction_dir=result_dir/'final_result/data'
    assert {p.stem for p in prediction_dir.glob('*.txt')}==set(ids)
    communication=None
    if fold and not args.diagnostic_holdout:
        communication_path=result_dir/'communication_rank0.jsonl'
        accounts=[json.loads(line) for line in communication_path.read_text().splitlines()]
        assert [row['frame_id'][0] for row in accounts]==ids
        for row in accounts:
            assert row['channel']==('identity' if args.identity_codec_diagnostic else 'awgn') and row['snr_db']==10.0
            assert row['pilot_complex_uses']==row['header_complex_uses']==0
            assert row['data_complex_uses']==row['total_complex_uses']>0
            assert len(row['tx_energy_per_frame'])==1
            assert np.isfinite(row['tx_energy_per_frame']).all()
            assert abs(row['tx_energy_per_frame'][0]/row['total_complex_uses']-1)<=1e-4
            assert row['allocation'] in ['uniform','geometry_task','geometry','task']
            if args.stereo_feature_link:
                assert row['allocation']=='uniform' and row['boundary']=='stereo_features_before_receiver_cost'
                assert row['total_complex_uses']==62400 and row['stereo_complex_uses']==49920 and row['appearance_complex_uses']==12480
                assert abs(row['cbr_complex_per_input_real_scalar']-1/38.4)<1e-12
        communication=dict(sha256=sha256(communication_path),frames=len(accounts),
            channel=sorted(set(row['channel'] for row in accounts)),
            allocation=sorted(set(row['allocation'] for row in accounts)),
            complex_uses=sorted(set(row['total_complex_uses'] for row in accounts)),
            CBR=sorted(set(row['cbr_complex_per_input_real_scalar'] for row in accounts)))
    files={}
    for frame,anno,info in zip(ids,annos,infos):
        path=prediction_dir/(frame+'.txt')
        audit_prediction(anno,path)
        label=root/'training/label_2'/(frame+'.txt')
        rows=[line.split() for line in label.read_text().splitlines()]
        gt=info['annos']
        assert [x[0] for x in rows]==gt['name'].tolist()
        for index,row in enumerate(rows):
            values=np.asarray([float(x) for x in row[3:15]])
            expected=np.r_[gt['alpha'][index],gt['bbox'][index],
                gt['dimensions'][index][[1,2,0]],gt['location'][index],gt['rotation_y'][index]]
            np.testing.assert_allclose(values,expected,rtol=1e-6,atol=1e-5)
        files[frame]=dict(prediction_sha256=sha256(path),label_sha256=sha256(label),
            calibration_sha256=sha256(root/'training/calib'/(frame+'.txt')),
            predictions=len(anno['name']))
    from liga.datasets.kitti.kitti_object_eval_python import eval as evaluator
    _,recomputed=evaluator.get_official_eval_result([x['annos'] for x in infos],annos,
                                                  ['Car','Pedestrian','Cyclist'])
    assert recomputed
    for key,value in recomputed.items():
        assert np.isfinite(value) and 0<=value<=100
        assert key in metrics and abs(metrics[key]-value)<=1e-6,(key,value,metrics.get(key))
    result=dict(state='passed',evidence_type=('identity_codec_holdout_diagnostic_AP_audit' if args.identity_codec_diagnostic else 'uncompressed_holdout_diagnostic_AP_audit' if args.diagnostic_holdout else 'training_only_holdout_artifact_and_recomputed_AP_audit' if fold else 'full_LIGA_KITTI_artifact_and_recomputed_AP_audit'),
        frames=len(ids),predictions=sum(x['predictions'] for x in files.values()),
        empty_frames=sum(x['predictions']==0 for x in files.values()),
        run_manifest_sha256=sha256(args.manifest),source_manifest_sha256=sha256(args.source_manifest),
        preparation_sha256=sha256(args.preparation),metrics_sha256=sha256(metrics_path),
        result_pickle_sha256=sha256(result_dir/'result.pkl'),
        all_file_hashes_sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
        recomputed_metrics=recomputed,files=files,
        tuning_fold_sha256=sha256(args.tuning_fold) if fold else None,
        communication=communication,
        limitations=('fresh prediction/label/calibration hashes and raw-GT versus infos checks; image/LiDAR identity comes from verified archives; not original communication codec reproduction'
                     + ('; codec with zero channel noise, not physical wireless performance' if args.identity_codec_diagnostic else '')
                     + ('; original detector pretraining included this codec holdout; not main-validation scientific result' if fold else '')))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,default=serializable,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files'},default=serializable))


if __name__=='__main__':
    main()
