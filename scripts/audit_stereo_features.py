#!/usr/bin/env python3
"""Validate independently saved feature reductions, coverage and F5 receiver boundary."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import yaml


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--AP-audit', dest='ap_audit', type=Path, required=True)
    parser.add_argument('--fold', type=Path, required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-channel', choices=['identity','awgn'], default='identity')
    parser.add_argument('--checkpoint-sha256', required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('preserve earlier audit; choose unique output')
    run = json.loads(args.manifest.read_text())
    ap = json.loads(args.ap_audit.read_text())
    assert run['state']=='finished' and run['mode']=='test' and ap['state']=='passed'
    assert ap['evidence_type']==('identity_codec_holdout_diagnostic_AP_audit' if args.expected_channel=='identity'
                                else 'training_only_holdout_artifact_and_recomputed_AP_audit')
    assert ap['run_manifest_sha256']==digest(args.manifest)
    cfg_path=Path(run['arguments'][run['arguments'].index('--cfg_file')+1])
    assert digest(cfg_path)==run['config_sha256']
    backbone=yaml.safe_load(cfg_path.read_text())['MODEL']['BACKBONE_3D']
    link=backbone['STEREO_FEATURE_LINK']
    assert link['enabled'] and link['channel']==args.expected_channel and link['snr_db']==10.
    assert not backbone['SEMANTIC_LINK']['enabled'] and backbone['STUDENT_ENCODER']['enabled']
    assert run['run_checkpoint_sha256']==args.checkpoint_sha256
    fold=json.loads(args.fold.read_text())
    rows=[json.loads(line) for line in args.features.read_text().splitlines()]
    assert [row['frame_id'] for row in rows]==fold['folds']['geocomm_tune_holdout']['ids']
    assert len(rows)==ap['frames']==372
    summaries={}
    for branch in ('left_stereo','right_stereo','appearance'):
        squared_error=squared_reference=dot=squared_received=0.
        count=0
        for row in rows:
            assert row['channel']==args.expected_channel
            r=row[branch];s=r['sums'];n=math.prod(r['shape'])
            assert r['elements']==n>0 and r['shape'][0]==1
            assert all(math.isfinite(value) for value in s.values())
            assert min(s['reference_squared'],s['received_squared'],s['error_squared'])>=0
            assert 0<=s['reference_negative']<=n and 0<=s['received_negative']<=n
            for key,value in [('mse',s['error_squared']/n),
                              ('reference_rms',math.sqrt(s['reference_squared']/n)),
                              ('received_rms',math.sqrt(s['received_squared']/n)),
                              ('reference_mean',s['reference_sum']/n),
                              ('received_mean',s['received_sum']/n),
                              ('reference_negative_fraction',s['reference_negative']/n),
                              ('received_negative_fraction',s['received_negative']/n)]:
                assert math.isclose(r[key],value,rel_tol=1e-12,abs_tol=1e-12),(branch,key)
            expected_nmse=s['error_squared']/s['reference_squared'] if s['reference_squared'] else None
            denom=math.sqrt(s['reference_squared']*s['received_squared'])
            expected_cos=s['dot']/denom if denom else None
            assert r['nmse']==expected_nmse and r['cosine']==expected_cos
            if expected_cos is not None: assert -1-1e-10<=expected_cos<=1+1e-10
            assert r['shape']==([1,32,80,312] if branch=='appearance' else [1,32,320,1248])
            squared_error+=s['error_squared'];squared_reference+=s['reference_squared']
            squared_received+=s['received_squared'];dot+=s['dot'];count+=n
        summaries[branch]=dict(elements=count, pooled_over_frames_mse=squared_error/count,
            pooled_over_frames_nmse=squared_error/squared_reference if squared_reference else None,
            pooled_over_frames_cosine=dot/math.sqrt(squared_reference*squared_received) if squared_reference*squared_received else None,
            median_frame_nmse=statistics.median(r[branch]['nmse'] for r in rows if r[branch]['nmse'] is not None),
            median_frame_cosine=statistics.median(r[branch]['cosine'] for r in rows if r[branch]['cosine'] is not None),
            reference_shapes=sorted(set(tuple(r[branch]['shape']) for r in rows)))
    report=json.loads(args.report.read_text())
    assert report['state']=='finished' and report['frames']==372 and report['channel']==args.expected_channel
    assert report['records_sha256']==digest(args.features) and report['checkpoint_sha256']==args.checkpoint_sha256
    assert report['calls']==dict(frames=372,student=744,codec=372,channel=372,build_cost=372,forbidden=0)
    loading=report['checkpoint_loading']
    assert loading['required_states']==535 and len(loading['state_hashes'])==535
    assert loading['state_hashes']==report['final_state_hashes']
    assert report['project_sources']==run['project_sources']
    for row in rows:
        assert row['sequence']==['student','student','link_start','channel','link_done','build_cost']
        assert row['cost_inputs_are_received_features'] and row['appearance_input_is_received_feature']
        assert not row['autograd_enabled'] and set(row['sensor_input_keys'])<=set(['batch_size','left_img','right_img','calib','image_shape','frame_id'])
        account=row['accounting']
        assert account['channel']==args.expected_channel and account['snr_db']==10. and account['allocation']=='uniform'
        assert account['boundary']=='stereo_features_before_receiver_cost'
        assert account['total_complex_uses']==account['data_complex_uses']==62400
        assert account['stereo_complex_uses']==49920 and account['appearance_complex_uses']==12480
        assert account['pilot_complex_uses']==account['header_complex_uses']==0
        assert len(account['tx_energy_per_frame'])==1 and math.isfinite(account['tx_energy_per_frame'][0])
        assert abs(account['tx_energy_per_frame'][0]/62400-1)<=1e-4
        assert math.isclose(account['cbr_complex_per_input_real_scalar'],1/38.4,rel_tol=1e-12)
    assert ap['communication']['complex_uses']==[62400]
    record=dict(state='passed',scope='feature reductions/coverage/geometry audit; raw tensors not retained or independently rerun',
        frames=len(rows), summaries=summaries, feature_records_sha256=digest(args.features),
        run_manifest_sha256=digest(args.manifest),AP_audit_sha256=digest(args.ap_audit),
        checkpoint_sha256=run['run_checkpoint_sha256'],boundary_report_sha256=digest(args.report),
        readonly_states=535,executed_calls=report['calls'],source_sha256=digest(__file__),
        limitations='passive same-input observation with saved reductions, not fresh raw-feature recomputation; '
                    'original detector pretraining included heldout IDs; not a wireless gain or unique failure-cause proof')
    args.output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(record))


if __name__=='__main__':
    main()
