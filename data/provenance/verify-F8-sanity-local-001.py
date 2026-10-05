"""Independent local check of the complete twelve-update engineering evidence."""
import hashlib,json,math,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];PREFIX='stereo-encoder-native-sanity-001'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    cp=ROOT/'data/provenance'/f'{PREFIX}-closure.json';c=read(cp)
    assert c['state']=='closed_all_audits_passed' and c['engineering'] and c['training_updates_per_arm']==6 and not c['evaluations']
    for p,h in c['artifacts_sha256'].items():assert sha(ROOT/p)==h
    cycle=read(ROOT/'data/runs'/f'{PREFIX}-cycle.json');launch=read(ROOT/'data/runs'/f'{PREFIX}-launch.json')
    assert cycle['pid']==launch['pid'] and cycle['state']=='finished_all_independent_audits_passed'
    roots={'project':ROOT,'liga':ROOT/'third_party/LIGA-Stereo','mmdet':ROOT/'third_party/mmdetection_kitti','stereo_rcnn':ROOT/'third_party/Stereo-RCNN','F7':ROOT,'experiment':ROOT}
    sources=read(ROOT/'data/provenance'/f'server-source-manifest-{PREFIX}.json')
    for key,value in sources.items():
        expected=value['actual_sources'];files={p:sha(roots[key]/p) for p in expected['file_hashes']}
        assert files==expected['file_hashes'] and hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()==expected['sha256']==c['source_sha256'][key]
    pair=read(ROOT/'data/runs'/f'{PREFIX}-pair-audit.json');assert pair['steps']==6 and pair['state']=='passed' and sha(ROOT/'data/runs'/f'{PREFIX}-pair-audit.json')==c['paired_audit_sha256']
    rows={};runs={};empty={}
    for arm,n,fixed in [('codec',16,519),('joint',51,484)]:
        mp=ROOT/'data/runs'/f'{PREFIX}-{arm}.json';m=read(mp);ap=ROOT/'data/runs'/f'{PREFIX}-{arm}-training-audit.json';a=read(ap)
        assert m['state']=='finished' and a['state']=='passed' and a['manifest_sha256']==sha(mp)
        assert a['steps']==m['optimizer_steps']==a['all_optimizer_update_counts']==6
        assert a['trained_parameter_tensors']==n and a['inactive_states_identical']==m['frozen_states_identical']==fixed
        assert a['checkpoint_sha256']==m['checkpoint_sha256']==pair['arms'][arm]['checkpoint_sha256']
        assert m['initialization_sha256']==pair['parent_sha256']=='ea93fb1ec7d7328e0ae603ec40385882da255414e876efa2c1abda4fa4de36a3'
        assert a['noise_rng_replay']['state']=='passed' and a['noise_rng_replay']['actual_calls']==6
        raw=ap.with_suffix('.records')/'training.jsonl';assert sha(raw)==a['raw_snapshot_sha256']==m['training_records_sha256']
        rows[arm]=[json.loads(x) for x in raw.read_text().splitlines()];assert len(rows[arm])==6;empty[arm]=0;runs[arm]=m
    for key in ('initialization_sha256','full_initialization_sha256','protocol_sha256','config_sha256','source_identities','dataset','schedule'):assert runs['codec'][key]==runs['joint'][key]
    assert set(runs['codec']['trainable_parameter_names'])<set(runs['joint']['trainable_parameter_names'])
    generator=random.Random(1717);prior=None
    for i,(a,b) in enumerate(zip(rows['codec'],rows['joint']),1):
        snr=generator.uniform(0,20);assert a['augmented_evidence']==b['augmented_evidence'] and a['channel_condition']==b['channel_condition']
        fp=a['augmented_evidence'];assert hashlib.sha256(json.dumps({k:v for k,v in fp.items() if k!='sha256'},sort_keys=True,separators=(',',':')).encode()).hexdigest()==fp['sha256']
        cond=a['channel_condition'];assert cond['step']==i and cond['snr_db']==snr and cond['noise_seed']==1718
        if prior is not None:assert cond['noise_rng_before_sha256']==prior
        assert cond['noise_rng_before_sha256']!=cond['noise_rng_after_sha256'];prior=cond['noise_rng_after_sha256']
        for arm,row in [('codec',a),('joint',b)]:
            assert row['step']==i and row['optimization_scope']==arm and row['channel']=='awgn' and row['snr_db']==snr
            assert row['GT_introduced_only_at_head'] and row['cost_inputs_are_received_features'] and row['appearance_input_is_received_feature'] and row['all_modules_eval']
            assert row['total_complex_uses']==62400 and row['pilot_complex_uses']==row['header_complex_uses']==0 and abs(row['tx_energy']/62400-1)<1e-4
            assert math.isfinite(row['loss']) and math.isclose(row['loss'],row['classification_loss']+row['location_loss']+row['direction_loss']+row['iou_loss_raw']*row['iou_weight'],rel_tol=3e-6,abs_tol=1e-6)
            assert set(row['parameter_gradients'])==set(runs[arm]['trainable_parameter_names']) and all(v['finite'] and math.isfinite(v['norm']) for v in row['parameter_gradients'].values())
            if row['empty_GT']:
                empty[arm]+=1;assert row['GT_shape']==[1,0,8] and row['positive_anchors']==0 and row['background_anchors']>0
                assert row['location_loss']==row['direction_loss']==row['iou_loss_raw']==0
    assert empty['codec']==empty['joint']>0
    result=dict(state='passed',closure_sha256=sha(cp),verifier_sha256=sha(Path(__file__)),paired_steps=6,empty_GT=empty,artifacts_verified=len(c['artifacts_sha256']),scope='All twelve raw metadata rows, source bytes and sealed full server native state/Adam/noise audits checked locally; full tensors/noise replay and actual terminal check executed on sheng. No AP result.')
    out=ROOT/'data/provenance'/f'{PREFIX}-local-verification-001.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
