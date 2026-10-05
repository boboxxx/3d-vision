"""Require sealed predecessor results and actual terminal native diagnostics."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def verify_eligibility():
    paths={}
    cp=ROOT/'data/provenance/stereo-channel-native-seed17-001-closure.json';c=read(cp)
    assert c['state']=='closed_all_audits_passed' and not c['engineering'] and c['training_updates_per_arm']==3340 and len(c['evaluations'])==4
    for name,h in c['artifacts_sha256'].items():assert sha(ROOT/name)==h
    lp=ROOT/'data/provenance/stereo-channel-native-seed17-001-local-verification-001.json';local=read(lp)
    assert local['state']=='passed' and local['closure_sha256']==sha(cp) and local['paired_steps']==3340
    paths.update({str(cp.relative_to(ROOT)):sha(cp),str(lp.relative_to(ROOT)):sha(lp)})
    prefix='geometry-risk-antithetic-001';mp=ROOT/'data/runs'/f'{prefix}.json';m=read(mp)
    assert m['state']=='finished8_paired_observations_independent_audit_pending' and m['completed_native_passes']==8208
    try:os.kill(m['pid'],0)
    except ProcessLookupError:pass
    else:raise RuntimeError('preceding native process still exists')
    ap=ROOT/'data/provenance'/f'{prefix}-native-audit.json';a=read(ap)
    assert a['state']=='passed_full8_independent_antithetic_native_audit' and a['manifest_sha256']==sha(mp)
    paths.update({str(mp.relative_to(ROOT)):sha(mp),str(ap.relative_to(ROOT)):sha(ap)})
    result=ROOT/'data/analysis'/f'{prefix}-frozen-predictors-001.json'
    for suffix in ('statistics-audit','statistics-local-audit-001'):
        p=ROOT/'data/provenance'/f'{prefix}-{suffix}.json';v=read(p)
        assert v['state']=='passed_independent_fresh8_frozen_predictor_statistics' and v['result_sha256']==sha(result) and v['native_audit_sha256']==sha(ap)
        paths[str(p.relative_to(ROOT))]=sha(p)
    p=ROOT/'data/provenance'/f'{prefix}-transferred-verification-001.json';v=read(p)
    assert v['state']=='passed_complete_transferred_raw_rows_and_reports' and v['native_rows']==8208 and v['native_audit_sha256']==sha(ap)
    paths[str(p.relative_to(ROOT))]=sha(p)
    return dict(state='passed_predecessor_closures',artifacts_sha256=paths,actual_terminal_PID=m['pid'])
