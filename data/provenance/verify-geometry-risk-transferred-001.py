"""Transferred 260 raw rows/metadata audit; large arrays replayed on sheng only."""
import hashlib,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
PREFIX='geometry-risk-native-engineering-001'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())


def main():
    manifest=ROOT/'data/engineering'/f'{PREFIX}.json';run=read(manifest)
    apath=ROOT/'data/provenance'/f'{PREFIX}-audit.json';audit=read(apath)
    assert run['state']=='finished_native_engineering_audit_pending' and audit['state']=='passed_independent_native_engineering_audit'
    assert audit['manifest_sha256']==sha(manifest) and audit['verifier_sha256']==sha(ROOT/'data/provenance/verify-geometry-risk-native-engineering-001.py')
    assert audit['actual_terminal_PIDs']==[73911,68491] and run['pid']==73911
    qpath=ROOT/'data/runs'/f'{PREFIX}-queue.json';queue=read(qpath)
    assert queue['state']=='native_engineering_terminal_independent_audit_pending' and sha(qpath)==audit['queue_manifest_sha256']
    assert run['readonly_states']==535 and run['optimizer_updates']==0 and run['no_parameter_gradients']
    assert run['sole_parent_sha256']=='77bc73021a139099e3d0f9c12e0fba8b17eb62f9b9c243eb8f0ad099bd6e8867'
    assert run['source_identities']==run['source_identities_after']==queue['source_identities']
    roots={'project':ROOT,'liga':ROOT/'third_party/LIGA-Stereo','mmdet':ROOT/'third_party/mmdetection_kitti',
           'stereo_rcnn':ROOT/'third_party/Stereo-RCNN','F7':ROOT,'final_evaluation':ROOT,'experiment':ROOT}
    for key,item in run['source_identities'].items():
        files={p:sha(roots[key]/p) for p in item['file_hashes']};assert files==item['file_hashes']
        assert hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()==item['sha256']
    for p,h in run['metadata_identities'].items():assert sha(ROOT/p)==h
    assert run['CPU_gate_sha256']==sha(ROOT/'data/engineering/geometry-risk-native-sheng-CPU-001.json')
    directory=ROOT/'data/engineering'/f'{PREFIX}-records';raw=directory/'native-loss.jsonl'
    assert sha(raw)==audit['records_sha256']==run['records_sha256']
    rows=[json.loads(line) for line in raw.read_text().splitlines()];assert len(rows)==audit['native_rows']==260
    totalenergy=0.;summaries={}
    for i,frame in enumerate(['000000','000003']):
        reportpath=directory/(frame+'.json');r=read(reportpath)
        assert sha(reportpath)==run['frames'][frame]['report_sha256']==audit['summaries'][frame]['frame_report_sha256']
        assert r['fixture_sha256']==audit['summaries'][frame]['fixture_sha256']
        assert r['state_hashes_before']==r['state_hashes_after']==run['all535_final_state_hashes'] and len(r['state_hashes_before'])==535
        assert r['native_loss_passes']==130 and r['attempted_complex_uses']==130*62400 and r['no_parameter_gradients']
        selected=rows[i*130:(i+1)*130];tx=None
        for j,row in enumerate(selected):
            assert row['frame_id']==frame and row['GT_introduced_only_at_head'] and row['cost_inputs_are_received_features'] and row['appearance_input_is_received_feature']
            assert row['all_modules_eval'] and row['total_complex_uses']==62400 and row['pilot_complex_uses']==row['header_complex_uses']==0
            assert math.isfinite(row['loss']) and math.isclose(row['loss'],row['classification_loss']+row['location_loss']+row['direction_loss']+row['iou_loss_raw']*row['iou_weight'],rel_tol=3e-6,abs_tol=1e-6)
            cond=row['condition'];assert cond['global_rng_before']==cond['global_rng_after'] and cond['consumed']
            if tx is None:tx=cond['transmitted_symbols_sha256']
            assert tx==cond['transmitted_symbols_sha256'] and abs(row['tx_energy']/62400-1)<1e-4;totalenergy+=row['tx_energy']
            if j==0:
                assert cond['mode']=='clean' and row['loss']==r['clean_loss'] and len(row['gradient_norms'])==5
                assert cond['PCG64']['before']==cond['PCG64']['after'] and cond['transmitted_symbols_sha256']==cond['received_symbols_sha256']
            elif j==129:assert cond['mode']=='full_awgn' and row['loss']==r['full_AWGN10_loss']
            else:
                assert cond['mode']=='group' and cond['group']==(j-1)//4 and cond['draw']==(j-1)%4 and not row['gradient_norms']
            if j:assert row['loss_minus_clean']==row['loss']-r['clean_loss']
        for group,item in enumerate(r['damage']):
            delta=[x['loss']-r['clean_loss'] for x in selected[1+4*group:5+4*group]]
            assert item['draw_damage']==delta and item['signed_damage_mean']==float(np.mean(delta))
            assert item['positive_damage_mean']==max(float(np.mean(delta)),0.) and item['sample_variance']==float(np.var(delta,ddof=1))
        summaries[frame]={k:r[k] for k in ['clean_loss','full_AWGN10_loss','native_loss_passes','attempted_complex_uses','valid_image_shape']}
    assert totalenergy==audit['actual_total_transmit_energy'] and run['attempted_complex_uses']==audit['attempted_complex_uses']==16224000
    assert run['calls']==dict(steps=260,student=520,codec=260,channel=260,build_cost=260,map_to_bev=260,BEV=260,head3D=260,forbidden=0)
    output=ROOT/'data/provenance'/f'{PREFIX}-transferred-verification-001.json'
    result=dict(state='passed_260_transferred_raw_rows_and_sealed_native_audit',manifest_sha256=sha(manifest),audit_sha256=sha(apath),
                raw_records_sha256=sha(raw),verifier_sha256=sha(Path(__file__)),native_rows=260,summaries=summaries,
                scope='Complete local raw scalar/source/state metadata verification; complete sensor/noise/feature/posterior/parent replay performed independently on sheng. No fit/AP/allocation claim.')
    with output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'state':result['state'],'native_rows':260}))


if __name__=='__main__':main()
