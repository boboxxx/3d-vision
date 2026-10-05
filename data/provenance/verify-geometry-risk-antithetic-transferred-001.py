"""Verify complete transferred raw rows/reports; large-array replay remains on sheng."""
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PREFIX='geometry-risk-antithetic-001'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    runpath=ROOT/'data/runs'/f'{PREFIX}.json';run=read(runpath)
    auditpath=ROOT/'data/provenance'/f'{PREFIX}-native-audit.json';audit=read(auditpath)
    assert run['state']=='finished8_paired_observations_independent_audit_pending'
    assert audit['state']=='passed_full8_independent_antithetic_native_audit'
    assert audit['manifest_sha256']==sha(runpath) and audit['verifier_sha256']==sha(Path(__file__).with_name('verify-geometry-risk-antithetic-001.py'))
    assert audit['helper_sha256']==sha(Path(__file__).with_name('verify-geometry-risk-pilot-001.py'))
    launch=ROOT/'data/runs'/f'{PREFIX}-launch.json'
    assert sha(launch)==audit['launch_sha256'] and read(launch)['pid']==run['pid']==audit['actual_terminal_PID']
    assert sha(ROOT/'logs'/f'{PREFIX}.log')==audit['native_log_sha256']
    roots={'project':ROOT,'liga':ROOT/'third_party/LIGA-Stereo','mmdet':ROOT/'third_party/mmdetection_kitti',
           'stereo_rcnn':ROOT/'third_party/Stereo-RCNN','F7':ROOT,'final_evaluation':ROOT,'experiment':ROOT,'pilot':ROOT,'antithetic':ROOT}
    assert run['source_identities']==run['source_identities_after']
    for key,item in run['source_identities'].items():
        files={p:sha(roots[key]/p) for p in item['file_hashes']}
        assert files==item['file_hashes']
        assert hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()==item['sha256']
    for p,h in run['metadata_identities'].items():assert sha(ROOT/p)==h
    directory=ROOT/'data/engineering'/f'{PREFIX}-records'
    raw=directory/'native-loss.jsonl'
    assert sha(raw)==run['records_sha256']==audit['records_sha256']
    rows=[json.loads(x) for x in raw.read_text().splitlines()];assert len(rows)==8208
    ids=run['ids'];assert len(ids)==len(set(ids))==8
    rngstate=run['initial_PCG64_state'];energy=0.;artifacts={str(raw.relative_to(ROOT)):sha(raw)}
    for i,fid in enumerate(ids):
        path=directory/(fid+'.json');report=read(path)
        assert sha(path)==run['frames'][fid]['report_sha256']==audit['summaries'][fid]['frame_report_sha256']
        assert report['state_hashes_before']==report['state_hashes_after']==run['all535_final_state_hashes']
        assert len(report['state_hashes_before'])==535 and report['no_parameter_gradients']
        frame_rows=rows[i*1026:(i+1)*1026]
        assert frame_rows[0]['loss']==report['clean_loss'] and frame_rows[-1]['loss']==report['full_AWGN10_loss']
        for j,row in enumerate(frame_rows):
            c=row['condition'];mode='clean' if j==0 else 'full_awgn' if j==1025 else 'group'
            assert row['frame_id']==c['frame_id']==fid and c['attempt']==i*1026+j+1
            assert c['mode']==mode and c['consumed'] and c['PCG64']['before']==rngstate
            rngstate=c['PCG64']['after']
            assert c['global_rng_before']==c['global_rng_after']
            if mode=='group':
                assert c['group']==(j-1)//32 and c['pair']==c['draw']==((j-1)//2)%16 and c['sign']==(1 if j%2 else -1)
            assert row['total_complex_uses']==62400 and row['pilot_complex_uses']==row['header_complex_uses']==0
            assert row['GT_introduced_only_at_head'] and row['cost_inputs_are_received_features'] and row['appearance_input_is_received_feature'] and row['all_modules_eval']
            assert math.isfinite(row['loss']) and math.isclose(row['loss'],row['classification_loss']+row['location_loss']+row['direction_loss']+row['iou_loss_raw']*row['iou_weight'],rel_tol=3e-6,abs_tol=1e-6)
            if j==0:assert set(row['gradient_norms'])=={'symbols','native_cost','left_stereo','right_stereo','appearance'}
            else:assert not row['gradient_norms'] and row['loss_minus_clean']==row['loss']-report['clean_loss']
            assert abs(row['tx_energy']/62400-1)<1e-4;energy+=row['tx_energy']
        for group,item in enumerate(report['damage']):
            selected=frame_rows[1+group*32:33+group*32]
            positive=[r['loss'] for r in selected[::2]];negative=[r['loss'] for r in selected[1::2]]
            dp=[v-report['clean_loss'] for v in positive];dn=[v-report['clean_loss'] for v in negative]
            q=[(p+n)/2 for p,n in zip(dp,dn)];odd=[(p-n)/2 for p,n in zip(dp,dn)]
            linear=[r['linear_gradient_dot_noise'] for r in selected[::2]]
            assert [-v for v in linear]==[r['linear_gradient_dot_noise'] for r in selected[1::2]]
            for key,expected in dict(positive_loss=positive,negative_loss=negative,positive_damage=dp,negative_damage=dn,symmetric=q,antisymmetric=odd,linear=linear,antisymmetric_minus_linear=[a-b for a,b in zip(odd,linear)]).items():assert item[key]==expected
            qm=sum(q)/16;sm=sum(odd)/16;lm=sum(linear)/16
            assert item['signed_symmetric_mean']==qm and item['positive_symmetric_mean']==max(qm,0.)
            assert math.isclose(item['antisymmetric_mean'],sm,rel_tol=1e-12,abs_tol=1e-15)
            for key,values,mean in [('symmetric_sample_variance',q,qm),('antisymmetric_sample_variance',odd,sm),('linear_sample_variance',linear,lm)]:
                assert math.isclose(item[key],sum((v-mean)**2 for v in values)/15,rel_tol=1e-12,abs_tol=1e-16)
        artifacts[str(path.relative_to(ROOT))]=sha(path)
    assert rngstate==run['final_PCG64_state']
    assert run['attempted_complex_uses']==audit['attempted_complex_uses']==8208*62400
    assert energy==audit['actual_total_transmit_energy']
    assert run['optimizer_updates']==0 and run['readonly_states']==535 and run['no_parameter_gradients']
    out=ROOT/'data/provenance'/f'{PREFIX}-transferred-verification-001.json';assert not out.exists()
    result=dict(state='passed_complete_transferred_raw_rows_and_reports',frames=8,native_rows=8208,
                verifier_sha256=sha(Path(__file__)),manifest_sha256=sha(runpath),native_audit_sha256=sha(auditpath),
                attempted_complex_uses=8208*62400,artifacts_sha256=artifacts,
                scope='Local complete raw metadata verification. Full sensors/features/posterior/noise arrays independently replayed on sheng only. Terminal proof inherited from sealed server audit.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'state':result['state'],'native_rows':8208,'frames':8}))

if __name__=='__main__':main()
