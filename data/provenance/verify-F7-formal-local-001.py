"""Verify sealed whole paired training and four endpoints without re-running AP."""
import hashlib
import json
import math
from pathlib import Path
import random

ROOT=Path(__file__).resolve().parents[2]
PREFIX='stereo-channel-native-seed17-001'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def main():
    closure_path=ROOT/'data/provenance'/f'{PREFIX}-closure.json'
    closure=read(closure_path)
    assert closure['state']=='closed_all_audits_passed' and not closure['engineering']
    assert closure['training_updates_per_arm']==3340 and closure['original21_unchanged']
    for path,value in closure['artifacts_sha256'].items():assert sha(ROOT/path)==value
    terminal=read(ROOT/'data/provenance'/f'{PREFIX}-actual-terminal-001.json')
    assert terminal['cycle_pid']==56655 and terminal['wrapper_pid']==56637
    assert terminal['cycle_actual_terminal'] and terminal['wrapper_actual_terminal']
    assert terminal['closure_sha256']==sha(closure_path)
    assert terminal['artifacts_sha256']==closure['artifacts_sha256']
    source=read(ROOT/'data/provenance'/f'server-source-manifest-{PREFIX}.json')
    source_roots={'project':ROOT,'liga':ROOT/'third_party/LIGA-Stereo',
                  'mmdet':ROOT/'third_party/mmdetection_kitti','stereo_rcnn':ROOT/'third_party/Stereo-RCNN','experiment':ROOT}
    for key,item in source.items():
        actual=item['actual_sources'];files={name:sha(source_roots[key]/name) for name in actual['file_hashes']}
        assert files==actual['file_hashes']
        payload=json.dumps(files,sort_keys=True,separators=(',',':')).encode()
        assert hashlib.sha256(payload).hexdigest()==actual['sha256']==closure['source_sha256'][key]
    pair=read(ROOT/'data/runs'/f'{PREFIX}-pair-audit.json')
    assert pair['state']=='passed' and pair['steps']==3340
    assert pair['parent_sha256']=='77bc73021a139099e3d0f9c12e0fba8b17eb62f9b9c243eb8f0ad099bd6e8867'
    runs={};rows={};empty={}
    for arm in ('identity','awgn'):
        run=read(ROOT/'data/runs'/f'{PREFIX}-{arm}.json');audit=read(ROOT/'data/runs'/f'{PREFIX}-{arm}-training-audit.json')
        assert run['state']=='finished' and audit['state']=='passed'
        assert run['optimizer_steps']==audit['steps']==audit['all_optimizer_update_counts']==3340
        assert audit['inactive_states_identical']==519 and audit['trained_parameter_tensors']==16
        assert audit['checkpoint_sha256']==run['checkpoint_sha256']==pair['arms'][arm]['checkpoint_sha256']
        assert audit['noise_rng_replay']['state']=='passed' and audit['noise_rng_replay']['actual_calls']==3340
        path=ROOT/'data/runs'/f'{PREFIX}-{arm}-training-audit.records/training.jsonl'
        assert sha(path)==audit['raw_snapshot_sha256']==run['training_records_sha256']
        runs[arm]=run;rows[arm]=[json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows[arm])==3340;empty[arm]=0
    for key in ('full_initialization_sha256','initialization_sha256','trainable_parameter_names','dataset','schedule','config_sha256','protocol_sha256'):
        assert runs['identity'][key]==runs['awgn'][key]
    rng=random.Random(1707);ids=[];previous={arm:None for arm in rows}
    for step,(a,b) in enumerate(zip(rows['identity'],rows['awgn']),1):
        snr=rng.uniform(0,20)
        assert a['step']==b['step']==step and a['frame_id']==b['frame_id']
        assert a['augmented_evidence']==b['augmented_evidence']
        fingerprint=a['augmented_evidence'];payload={k:v for k,v in fingerprint.items() if k!='sha256'}
        assert hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==fingerprint['sha256']
        ids.append(a['frame_id'])
        for arm,row in [('identity',a),('awgn',b)]:
            cond=row['channel_condition'];assert row['channel']==cond['channel']==arm
            assert row['snr_db']==cond['snr_db']==snr and cond['step']==step and cond['noise_seed']==1708
            assert cond['isolated_noise_generator'] and row['GT_introduced_only_at_head'] and row['all_modules_eval']
            assert row['cost_inputs_are_received_features'] and row['appearance_input_is_received_feature']
            assert (row['total_complex_uses'],row['stereo_complex_uses'],row['appearance_complex_uses'],row['pilot_complex_uses'],row['header_complex_uses'])==(62400,49920,12480,0,0)
            assert abs(row['tx_energy']/62400-1)<=1e-4
            assert all(math.isfinite(row[k]) for k in ('loss','preclip_grad_norm','classification_loss','location_loss','direction_loss'))
            if previous[arm] is not None:assert cond['noise_rng_before_sha256']==previous[arm]
            assert (cond['noise_rng_before_sha256']==cond['noise_rng_after_sha256'])==(arm=='identity')
            previous[arm]=cond['noise_rng_after_sha256'];empty[arm]+=row['empty_GT']
    fold=read(ROOT/'data/internal-tuning-fold-001.json')
    assert set(ids)==set(fold['folds']['geocomm_tune_train']['ids']) and len(set(ids))==3340
    assert empty=={'identity':62,'awgn':62}
    metrics={}
    assert len(closure['evaluations'])==4
    for rid,item in closure['evaluations'].items():
        ap=read(ROOT/'data/runs'/f'{rid}-AP-audit.json');feature=read(ROOT/'data/runs'/f'{rid}-feature-audit.json')
        boundary=read(ROOT/'data/runs'/f'{rid}-boundary-report.json');run=read(ROOT/'data/runs'/f'{rid}.json')
        assert ap['state']==feature['state']=='passed' and run['state']==boundary['state']=='finished'
        assert ap['frames']==feature['frames']==boundary['frames']==372 and feature['readonly_states']==535
        assert boundary['final_state_hashes']==boundary['checkpoint_loading']['state_hashes'] and len(boundary['final_state_hashes'])==535
        assert run['run_checkpoint_sha256']==item['checkpoint_sha256']==runs[item['arm']]['checkpoint_sha256']
        assert ap['communication']['complex_uses']==[62400] and ap['communication']['channel']==[item['channel']]
        assert set(ap['files'])==set(fold['folds']['geocomm_tune_holdout']['ids'])
        value={k:v for k,v in ap['recomputed_metrics'].items() if k.startswith('Car_3d/')}
        assert value==item['Car3D_AP_R40_percent'];metrics[rid]=value
    delta=metrics[PREFIX+'-awgn-test-awgn']['Car_3d/moderate_R40']-metrics[PREFIX+'-identity-test-awgn']['Car_3d/moderate_R40']
    assert delta==closure['primary_AWGN10_Moderate_treatment_minus_control_pp']
    output=ROOT/'data/provenance'/f'{PREFIX}-local-verification-001.json';assert not output.exists()
    output.write_text(json.dumps(dict(state='passed',closure_sha256=sha(closure_path),verifier_sha256=sha(Path(__file__)),
        actual_terminal_evidence_sha256=sha(ROOT/'data/provenance'/f'{PREFIX}-actual-terminal-001.json'),
        artifacts_verified=len(closure['artifacts_sha256']),paired_steps=3340,empty_GT=empty,metrics=metrics,
        primary_AWGN10_Moderate_treatment_minus_control_pp=delta,
        scope='Sealed paired raw records and full server audits; local audit does not re-run CUDA RNG, checkpoints or AP.'),indent=2)+'\n')
    print(json.dumps({'state':'passed','paired_steps':3340,'primary_delta_pp':delta}))


if __name__=='__main__':main()
