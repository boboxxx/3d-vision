#!/usr/bin/env python3
"""Validate independently saved feature reductions, coverage and slot geometry."""
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
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-channel', choices=['identity','awgn'], default='identity')
    parser.add_argument('--checkpoint-sha256', default='cb50b4d4073fb2e5be31533d949a2141a46389436a5080429753de8b038aef7c')
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
    link=yaml.safe_load(cfg_path.read_text())['MODEL']['BACKBONE_3D']['SEMANTIC_LINK']
    assert link['enabled'] and link['channel']==args.expected_channel and link['boundary']=='raw_cost'
    assert run['run_checkpoint_sha256']==args.checkpoint_sha256
    fold=json.loads(args.fold.read_text())
    rows=[json.loads(line) for line in args.features.read_text().splitlines()]
    assert [row['frame_id'] for row in rows]==fold['folds']['geocomm_tune_holdout']['ids']
    assert len(rows)==ap['frames']==372
    summaries={}
    for branch in ('cost','appearance','cost_pool_interpolate_reference','appearance_pool_interpolate_reference'):
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
            # All outputs share the exact same input reference for each branch.
            original=row['cost' if branch.startswith('cost') else 'appearance']
            assert r['shape']==original['shape'] and s['reference_squared']==original['sums']['reference_squared']
            squared_error+=s['error_squared'];squared_reference+=s['reference_squared']
            squared_received+=s['received_squared'];dot+=s['dot'];count+=n
        summaries[branch]=dict(elements=count, pooled_over_frames_mse=squared_error/count,
            pooled_over_frames_nmse=squared_error/squared_reference if squared_reference else None,
            pooled_over_frames_cosine=dot/math.sqrt(squared_reference*squared_received) if squared_reference*squared_received else None,
            median_frame_nmse=statistics.median(r[branch]['nmse'] for r in rows if r[branch]['nmse'] is not None),
            median_frame_cosine=statistics.median(r[branch]['cosine'] for r in rows if r[branch]['cosine'] is not None),
            reference_shapes=sorted(set(tuple(r[branch]['shape']) for r in rows)))
    for row in rows:
        for branch,strides in [('cost',link['geometry_stride']),('appearance',[link['appearance_stride']]*2)]:
            shape=row[branch]['shape']
            assert row[branch+'_pooled_shape']==shape[:2]+[(n+s-1)//s for n,s in zip(shape[2:],strides)]
        uses=link['complex_width']*(math.prod(row['cost_pooled_shape'][2:])+math.prod(row['appearance_pooled_shape'][2:]))
        assert ap['communication']['complex_uses']==[uses]
    record=dict(state='passed',scope='feature reductions/coverage/geometry audit; raw tensors not retained or independently rerun',
        frames=len(rows), summaries=summaries, feature_records_sha256=digest(args.features),
        run_manifest_sha256=digest(args.manifest),AP_audit_sha256=digest(args.ap_audit),
        checkpoint_sha256=run['run_checkpoint_sha256'],source_sha256=digest(__file__),
        limitations='passive same-input observation; pool/interpolate is not optimal reconstruction or AP ceiling; '
                    'original detector pretraining included heldout IDs; not a wireless gain or unique failure-cause proof')
    args.output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(record))


if __name__=='__main__':
    main()
