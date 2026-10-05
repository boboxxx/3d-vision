#!/usr/bin/env python3
"""Independent frozen checkpoint, native hook coverage and AP evidence audit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from geocomm.evidence import sha256,source_identity


def check(value,message):
    if not value: raise ValueError(message)


def audit_rows(rows,condition,ids):
    check([row['frame_id'] for row in rows]==ids,'complete ordered diagnostic frame coverage differs')
    check(condition in ('control','cost_only','appearance_only','both','depth_preserved'),'unknown condition')
    cost_pool=condition in ('cost_only','both','depth_preserved');app_pool=condition in ('appearance_only','both')
    cost_operation='adaptive_avg_pool_'+('1' if condition=='depth_preserved' else '8')+'_4_4_trilinear_restore' if cost_pool else 'identity'
    app_operation='adaptive_avg_pool_4_4_bilinear_restore' if app_pool else 'identity'
    for row in rows:
        check(row['condition']==condition and row['communication_enabled'] is False,'diagnostic operation scope differs')
        check(row['autograd_enabled'] is False,'automatic differentiation entered diagnostic inference')
        check(row['student_calls']==2 and row['raw_cost_calls']==1,'native call sequence differs')
        check(set(row['sensor_input_keys'])=={'batch_size','left_img','right_img','calib','image_shape','frame_id'},'sensor-only boundary differs')
        check(row['cost_shape']==[1,64,72,80,312] and row['appearance_shape']==[1,32,80,312],'native feature shape differs')
        check(row['cost_operation']==cost_operation and row['appearance_operation']==app_operation,'executed operation differs')
        check(row['cost_pooled_shape']==([1,64,72 if condition=='depth_preserved' else 9,20,78] if cost_pool else None),'cost grid differs')
        check(row['appearance_pooled_shape']==([1,32,20,78] if app_pool else None),'appearance grid differs')


def main():
    p=argparse.ArgumentParser()
    for name in ('manifest','report','records','AP-audit','checkpoint','config','protocol','fold','source-manifest','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();check(not args.output.exists(),'preserve prior evidence')
    run=json.loads(args.manifest.read_text());report=json.loads(args.report.read_text());ap=json.loads(args.AP_audit.read_text());fold=json.loads(args.fold.read_text())
    check(run['state']==report['state']=='finished' and ap['state']=='passed','inference/AP audit unfinished')
    check(run['mode']=='test' and run['seed']==run['rank_seed']==17 and report['scope']=='fixed clean-feature pooling diagnosis; no communication/training','inference seed/scope differs')
    check(sha256(args.checkpoint)==run['run_checkpoint_sha256']==report['checkpoint_sha256']=='984771355f145441f785a8bca5ddde2762fec4925d504298b7fb9abe8c3f57f0','fixedF4 checkpoint differs')
    check(sha256(args.config)==run['config_sha256'] and sha256(args.protocol)==run['protocol_sha256'],'fixed config/protocol differs')
    cfg=yaml.safe_load(args.config.read_text())['MODEL']['BACKBONE_3D']
    check(cfg['SEMANTIC_LINK']['enabled'] is False and cfg['STUDENT_ENCODER']['enabled'] and cfg['STUDENT_ENCODER']['allow_uncompressed_diagnostic'],'uncompressed student config differs')
    check(ap['evidence_type']=='uncompressed_holdout_diagnostic_AP_audit' and ap['frames']==372 and ap['run_manifest_sha256']==sha256(args.manifest) and ap['communication'] is None,'AP evidence identity/scope differs')
    check(report['frames']==372 and report['calls']==dict(frames=372,student=744,raw_cost=372,forbidden=0),'full native hook counts differ')
    check(report['records_sha256']==sha256(args.records),'raw pooling records identity differs')
    audit_rows([json.loads(line) for line in args.records.read_text().splitlines()],report['condition'],fold['folds']['geocomm_tune_holdout']['ids'])
    source=json.loads(args.source_manifest.read_text())
    for tree,key in [('project','project_sources'),('liga','detector_sources'),('mmdet','mmdet_sources')]:
        check(source[tree]['actual_sources']==run[key],'prelaunch source differs: '+tree)
    check(report['project_sources']==run['project_sources']==source_identity(ROOT,['src','scripts','configs','pyproject.toml']),'closure project source differs')
    loading=report['checkpoint_loading'];check(loading['required_states']==519,'required state count differs')
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)['model_state']
    unused=set(loading['unused_link_states']);expected_unused=set()
    for branch,indices in {'cost_encoder':[1,3],'cost_decoder':[0,2],'appearance_encoder':[1,3],'appearance_decoder':[0,2],'posterior':[0,2],'sensitivity_predictor.cost':[0,2],'sensitivity_predictor.appearance':[0,2]}.items():
        for index in indices:
            expected_unused.update('backbone_3d.semantic_link.'+branch+'.'+str(index)+'.'+field for field in ('weight','bias'))
    expected_unused.update('backbone_3d.semantic_link.generic_score.'+field for field in ('weight','bias'))
    check(len(checkpoint)==549 and unused==expected_unused and len(unused)==30 and unused<=set(checkpoint),'exact unused link states differ')
    required=set(checkpoint)-unused;hashes={}
    for name in required:
        value=checkpoint[name].detach().cpu().contiguous();check(torch.isfinite(value).all(),'nonfinite required checkpoint state')
        hashes[name]=hashlib.sha256(str(value.dtype).encode()+str(tuple(value.shape)).encode()+value.numpy().tobytes()).hexdigest()
    check(len(required)==519 and loading['state_hashes']==report['final_state_hashes']==hashes,'full required saved state differs before/after inference')
    result=dict(state='passed',scope='clean pooling intervention/full native AP audit; not communication/training',condition=report['condition'],frames=372,
        checkpoint_sha256=sha256(args.checkpoint),required_states_identical=519,unused_link_states=sorted(unused),
        manifest_sha256=sha256(args.manifest),report_sha256=sha256(args.report),records_sha256=sha256(args.records),
        AP_audit_sha256=sha256(args.AP_audit),source_manifest_sha256=sha256(args.source_manifest),
        Car_3d_AP_R40_percent={key:value for key,value in ap['recomputed_metrics'].items() if key.startswith('Car_3d/')},
        limitations='checkpoint/hook-source/operation-metadata and native AP evidence audit; no fresh raw pooling tensor recomputation, task representation sufficiency or unique failure-cause proof; detector author pretraining includes these heldout IDs')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__': main()
