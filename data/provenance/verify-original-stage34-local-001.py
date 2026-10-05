"""Verify synced completed-stage audit identities and all sealed raw records.

This supplements the server's full checkpoint/actual Adam audit; it does not
reload remote weights or turn reconstruction diagnostics into detection AP.
"""
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def check(value,message):
    if not value:raise RuntimeError(message)
fold=ROOT/'data/internal-tuning-fold-001.json'
ids=json.loads(fold.read_text())['folds']
original=json.loads((ROOT/'data/provenance/cao2025-staged-trainer-source-bfee9d8.json').read_text())
check(len(original)==21,'original source coverage')
for name,value in original.items():
    check(digest(ROOT/'reproduction/cao2025'/name)==value,'original source changed: '+name)
result={'state':'passed','scope':'local sealed-record verification supplements remote full-state Adam audits','stages':[]}
for stage,epochs in ((3,6),(4,10)):
    manifest=ROOT/f'data/runs/cao2025-full-native-seed17-001-stage{stage}.json'
    audit=manifest.with_name(manifest.stem+'-audit.json')
    run=json.loads(manifest.read_text());a=json.loads(audit.read_text())
    check(run['state']=='finished' and a['state']=='passed','stage not fully audited')
    check(a['manifest_sha256']==digest(manifest),'manifest identity')
    check(run['source_files']==a['source_files']==original,'source map')
    check(run['protocol_sha256']==digest(ROOT/'reproduction/cao2025/formal-protocol.md'),'protocol')
    check(run['fold_sha256']==digest(fold),'fold')
    check(run['samples']==run['optimizer_steps']==a['samples']==a['optimizer_updates']==3340*epochs,'budget')
    check(len(run['epochs'])==len(a['epochs'])==epochs,'epochs')
    raw=audit.with_suffix('.records')/'training.jsonl'
    check(digest(raw)==a['raw_snapshot_sha256']==run['training_records_sha256'],'sealed raw identity')
    rows=[json.loads(line) for line in raw.read_text().splitlines()]
    check(len(rows)==3340*epochs,'raw length')
    for index,row in enumerate(rows,1):
        check(row['sample']==row['optimizer_updates']==index and row['epoch']==(index-1)//3340+1,'raw continuity')
        check(row['updated'] and row['erasure'] is None and row['accounting'] is None and row['snr_db'] is None,'nonwireless scope')
        check(all(math.isfinite(row[k]) and row[k]>=0 for k in ('loss','preclip_grad_norm')),'finite loss/grad')
    for epoch in range(1,epochs+1):
        chunk=rows[(epoch-1)*3340:epoch*3340]
        check(len({r['frame_id'] for r in chunk})==3340 and {r['frame_id'] for r in chunk}==set(ids['geocomm_tune_train']['ids']),'epoch coverage')
        expected=12 if stage==3 else 638
        check(all(r['gradient_parameter_tensors']==expected for r in chunk),'gradient scope')
        summary=a['epochs'][epoch-1]
        check(summary['optimizer_states']==summary['active_parameter_tensors']==summary['changed_allowed_states']==expected,'remote actual Adam/state summary')
        check(summary['inactive_states_identical']==768-expected,'frozen coverage')
        check(summary['checkpoint_sha256']==run['epochs'][epoch-1]['checkpoint_sha256'],'checkpoint identity')
    held=audit.with_suffix('.records')/'final_holdout.jsonl'
    check(digest(held)==run['final_holdout']['records_sha256'],'holdout identity')
    heldrows=[json.loads(line) for line in held.read_text().splitlines()]
    check([r['frame_id'] for r in heldrows]==ids['geocomm_tune_holdout']['ids'],'holdout order')
    check(all(math.isfinite(r['loss']) and r['loss']>=0 and r['erasure'] is None for r in heldrows),'holdout loss')
    check(sum(r['loss'] for r in heldrows)/372==run['final_holdout']['mean_loss_on_successes'],'holdout reduction')
    if stage in (3,4):
        prev=manifest.with_name(f'cao2025-full-native-seed17-001-stage{stage-1}.json')
        preva=prev.with_name(prev.stem+'-audit.json')
        check(run['predecessor_manifest_sha256']==digest(prev) and run['predecessor_audit_sha256']==digest(preva),'chain identity')
        check(run['predecessor_checkpoint_sha256']==json.loads(preva.read_text())['checkpoint_sha256'],'parent weights')
    result['stages'].append(dict(stage=stage,epochs=epochs,samples=len(rows),manifest_sha256=digest(manifest),audit_sha256=digest(audit),training_snapshot_sha256=digest(raw),holdout_snapshot_sha256=digest(held),checkpoint_sha256=a['checkpoint_sha256']))
result['original21_sources']=original
result['limitations']='Remote independent audit loaded every complete checkpoint and actual optimizer; this local check verifies retained records/chain, no fresh GPU inference or AP.'
output=ROOT/'data/provenance/cao2025-full-native-seed17-001-stage34-local-verification.json'
check(not output.exists(),'preserve previous verification')
output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='original21_sources'}))
