"""Sealed150300-record stage5 check; supplements actual remote weight/Adam audit."""
import hashlib,json,math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def check(value,message):
    if not value:raise RuntimeError(message)


def layout(r):
    h,w=r['views'][0]['shape'];hp=(h+5)//6*6;wp=(w+5)//6*6;cells=[]
    for v in r['views']:
        mask=np.zeros((hp//2,wp//2),dtype=bool)
        for b in v['boxes']:
            x1,y1,x2,y2=b['xyxy'];mask[y1//2:(y2+1)//2,x1//2:(x2+1)//2]=1
        cells.append(int(mask.sum()))
    real=9*(2*(hp//6)*(wp//6)+sum(cells));data=(real+1)//2;control=1176+672*sum(len(v['boxes']) for v in r['views'])
    return dict(key_cells=cells,data_real_values=real,padding_real_values=real%2,data_uses=data,
                control_uses=control,pilot_uses=0,total_uses=data+control,cbr_complex_per_rgb_real_value=(data+control)/(6*h*w))


def wire(a,w):
    check(set(a)==set(w)|{'total_energy'},'accounting fields')
    for k,v in w.items():check(math.isclose(a[k],v,rel_tol=1e-12) if k.startswith('cbr') else a[k]==v,'layout '+k)
    check(math.isfinite(a['total_energy']) and abs(a['total_energy']/w['total_uses']-1)<=1e-4,'unit energy')


def main():
    output=ROOT/'data/provenance/cao2025-full-native-seed17-001-stage5-local-verification.json'
    check(not output.exists(),'unique verification')
    base=ROOT/'data/runs/cao2025-full-native-seed17-001-stage5'
    manifest=base.with_suffix('.json'); audit=Path(str(base)+'-audit.json')
    r=json.loads(manifest.read_text());a=json.loads(audit.read_text())
    check(r['state']=='finished' and a['state']=='passed' and r['stage']==a['stage']==5,'complete audited stage5')
    check(a['manifest_sha256']==digest(manifest),'manifest identity')
    source=json.loads((ROOT/'data/provenance/cao2025-staged-trainer-source-bfee9d8.json').read_text())
    check(len(source)==21 and r['source_files']==a['source_files']==source,'original21 source identity')
    for p,v in source.items():check(digest(ROOT/'reproduction/cao2025'/p)==v,'current original source')
    check(r['protocol_sha256']==digest(ROOT/'reproduction/cao2025/formal-protocol.md'),'protocol')
    fold=ROOT/'data/internal-tuning-fold-001.json';ids=json.loads(fold.read_text())['folds']
    check(r['fold_sha256']==digest(fold),'fold')
    train=ROOT/'data/engineering/cao2025-roi-train-001.jsonl'; held=ROOT/'data/engineering/cao2025-roi-holdout-001.jsonl'
    check(digest(train)==r['train_records_sha256'] and digest(held)==r['holdout_records_sha256'],'sensor records')
    native={x['frame_id']:x for x in map(json.loads,train.read_text().splitlines())}
    layouts={k:layout(v) for k,v in native.items()}
    check(set(native)==set(ids['geocomm_tune_train']['ids']) and len(native)==3340,'train scope')
    check(r['samples']==a['samples']==150300 and len(r['epochs'])==len(a['epochs'])==45,'budget')
    snapshots=audit.with_suffix('.records');raw=snapshots/'training.jsonl'
    check(digest(raw)==a['raw_snapshot_sha256']==r['training_records_sha256'],'sealed training snapshot')
    visited=set();updates=attempts=epoch_updates=erasures=empty_keys=0;epochs=[]
    with raw.open() as f:
        for line in f:
            row=json.loads(line);attempts+=1;epoch=(attempts-1)//3340+1
            check(row['sample']==attempts and row['epoch']==epoch,'attempt continuity')
            frame=row['frame_id'];check(frame in native and frame not in visited,'epoch coverage');visited.add(frame)
            check(row['shape']==[1,3,*native[frame]['views'][0]['shape']] and row['box_counts']==[len(v['boxes']) for v in native[frame]['views']],'native geometry')
            wire(row['accounting'],layouts[frame]);check(math.isfinite(row['snr_db']) and 6<=row['snr_db']<=18,'training SNR')
            check(row['loss_type']==('semantic_mse' if epoch<=25 else 'global_charbonnier'),'fixed25/20 phase')
            if row['erasure'] is None:
                check(row['updated'] and all(math.isfinite(row[k]) and row[k]>=0 for k in ['loss','preclip_grad_norm']),'actual finite update')
                # No key support means 24 key decoder parameters have no gradient in channel-only phase.
                absent=epoch<=25 and not sum(layouts[frame]['key_cells'])
                check(row['gradient_parameter_tensors']==(56 if absent else 80) if epoch<=25 else row['gradient_parameter_tensors']==718,'gradient scope')
                empty_keys+=int(absent);updates+=1;epoch_updates+=1
            else:
                check(isinstance(row['erasure'],str) and not row['updated'] and row['loss'] is None and row['preclip_grad_norm'] is None and row['gradient_parameter_tensors']==0,'charged erasure');erasures+=1
            check(row['optimizer_updates']==updates,'actual optimizer counter')
            if attempts%3340==0:
                check(visited==set(native),'whole epoch coverage')
                summary=a['epochs'][epoch-1];reported=r['epochs'][epoch-1]
                check(summary['samples']==reported['samples']==3340 and summary['updates']==reported['updates']==epoch_updates and summary['erasures']==reported['erasures']==erasures,'epoch attempt/update/erasure')
                check(summary['active_parameter_tensors']==(80 if epoch<=25 else 718),'epoch active states')
                check(summary['inactive_states_identical']==(640 if epoch<=25 else 2),'frozen states')
                check(summary['checkpoint_sha256']==reported['checkpoint_sha256'],'full checkpoint identity')
                epochs.append(dict(epoch=epoch,updates=epoch_updates,erasures=erasures,empty_ROI_key_decoder_absent=empty_keys))
                visited=set();epoch_updates=erasures=empty_keys=0
    check(attempts==150300 and not visited and updates==r['optimizer_steps']==a['optimizer_updates'],'full row/update coverage')
    hraw=snapshots/'final_holdout.jsonl';check(digest(hraw)==r['final_holdout']['records_sha256'],'holdout snapshot')
    hn={x['frame_id']:x for x in map(json.loads,held.read_text().splitlines())}
    hrows=[json.loads(x) for x in hraw.read_text().splitlines()]
    check([x['frame_id'] for x in hrows]==ids['geocomm_tune_holdout']['ids'],'complete holdout order')
    losses=[];erased=0
    for row in hrows:
        wire(row['accounting'],layout(hn[row['frame_id']]))
        if row['erasure'] is None:check(math.isfinite(row['loss']) and row['loss']>=0,'holdout loss');losses.append(row['loss'])
        else:check(isinstance(row['erasure'],str) and row['loss'] is None,'holdout erasure');erased+=1
    check(r['final_holdout']==dict(frames=372,erasures=erased,successful_frames=len(losses),mean_loss_on_successes=sum(losses)/len(losses) if losses else None,records_sha256=digest(hraw)),'holdout reduction')
    prev=base.with_name('cao2025-full-native-seed17-001-stage4').with_suffix('.json');preva=prev.with_name(prev.stem+'-audit.json')
    check(r['predecessor_manifest_sha256']==digest(prev) and r['predecessor_audit_sha256']==digest(preva),'stage4 chain')
    check(r['predecessor_checkpoint_sha256']==json.loads(preva.read_text())['checkpoint_sha256'],'stage4 weight identity')
    check(a['checkpoint_sha256']==r['epochs'][-1]['checkpoint_sha256'],'sole final epoch45')
    result=dict(state='passed',scope='sealed_record_chain_check_supplements_remote_all_weights_Adam_audit_NOT_AP',samples=attempts,optimizer_updates=updates,epochs=epochs,manifest_sha256=digest(manifest),audit_sha256=digest(audit),training_snapshot_sha256=digest(raw),holdout_snapshot_sha256=digest(hraw),checkpoint_sha256=a['checkpoint_sha256'],original21_sources=source)
    with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ['state','samples','optimizer_updates','checkpoint_sha256']}))


if __name__=='__main__':main()
