"""Require two independently audited native arms and paired actual input evidence."""
import argparse,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path[:0]=[str(HERE),str(ROOT/'src')]
from geocomm.evidence import sha256
from conditions import PARENT_SHA,PROTOCOL_SHA,CONFIG_SHA,STEPS,check,paired_records

def verified_arm(manifest,arm,steps):
    run=json.loads(manifest.read_text());audit_path=manifest.with_name(manifest.stem+'-training-audit.json');audit=json.loads(audit_path.read_text())
    check(run['state']=='finished' and audit['state']=='passed' and run['arm']==audit['arm']==arm,'finished independently audited native arm')
    check(audit['manifest_sha256']==sha256(manifest) and run['optimizer_steps']==audit['steps']==audit['all_optimizer_update_counts']==steps,'actual budget/audit identity')
    check(audit['inactive_states_identical']==519 and audit['trained_parameter_tensors']==16,'full state/optimizer scope')
    check(run['initialization_sha256']==PARENT_SHA and run['protocol_sha256']==PROTOCOL_SHA and run['config_sha256']==CONFIG_SHA,'parent/protocol/config')
    check(run['checkpoint_sha256']==audit['checkpoint_sha256']==sha256(Path(run['checkpoint_path'])),'final checkpoint identity')
    check(run['full_initialization_sha256']==audit['full_initialization_sha256']==sha256(Path(run['full_initialization_path'])),'full parent snapshot')
    sealed=audit_path.with_suffix('.records')/'training.jsonl';raw=Path(run['output_dir'])/'training.jsonl'
    check(sha256(raw)==sha256(sealed)==audit['raw_snapshot_sha256']==run['training_records_sha256'],'actual/sealed raw records')
    check(audit['source_manifest_sha256']==run['source_manifest_sha256'],'source input identity')
    check(audit['noise_rng_replay']['state']=='passed' and audit['noise_rng_replay']['actual_calls']==steps,'independent fixed noise-draw replay')
    return run,audit,sealed,audit_path

def main():
    p=argparse.ArgumentParser();p.add_argument('--identity-manifest',type=Path,required=True);p.add_argument('--awgn-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--engineering-sanity',action='store_true');args=p.parse_args()
    if args.output.exists():p.error('preserve previous pair evidence')
    steps=6 if args.engineering_sanity else STEPS;paths=[args.identity_manifest,args.awgn_manifest]
    arms=[verified_arm(path,arm,steps) for path,arm in zip(paths,('identity','awgn'))]
    clean,noisy=[v[0] for v in arms]
    expected_type='engineering_only' if args.engineering_sanity else 'exploratory_matched_channel_native3D_codec_adaptation'
    check(all(r['evidence_type']==a['evidence_type']==expected_type for r,a,_,_ in arms),'paired evidence scope')
    for key in ('seed','initialization_sha256','full_initialization_sha256','protocol_sha256','config_sha256','fold_sha256',
                'source_manifest_sha256','source_identities','trainable_parameter_names','dataset','schedule'):
        check(clean[key]==noisy[key],'paired condition differs: '+key)
    result=paired_records(arms[0][2],arms[1][2],steps)
    result.update(evidence_type=expected_type,parent_sha256=PARENT_SHA,
      full_initialization_sha256=clean['full_initialization_sha256'],source_manifest_sha256=clean['source_manifest_sha256'],
      arms={arm:dict(manifest_path=str(path.resolve()),manifest_sha256=sha256(path),audit_path=str(v[3].resolve()),audit_sha256=sha256(v[3]),
                    checkpoint_sha256=v[0]['checkpoint_sha256'],raw_snapshot_sha256=sha256(v[2])) for arm,path,v in zip(('identity','awgn'),paths,arms)})
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
