"""Terminal F7 all-artifact/fresh AP-file/source closure, no checkpoint selection."""
import argparse,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path[:0]=[str(HERE),str(ROOT/'src')]
from conditions import PARENT_SHA,check
from geocomm.evidence import sha256,source_identity
from train import source_specs

def read(path):return json.loads(Path(path).read_text())

def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);args=p.parse_args()
    cycle_path=ROOT/'data/runs'/f'{args.prefix}-cycle.json';cycle=read(cycle_path)
    closure=ROOT/'data/provenance'/f'{args.prefix}-closure.json';check(not closure.exists(),'preserve closure')
    check(cycle['prefix']==args.prefix and cycle['state']=='finished_all_independent_audits_passed','complete native cycle required')
    try:os.kill(cycle['pid'],0)
    except ProcessLookupError:pass
    else:raise RuntimeError('cycle PID stilllive; no source freeze release')
    source_path=ROOT/'data/provenance'/f'server-source-manifest-{args.prefix}.json';source=read(source_path)
    check(sha256(source_path)==cycle['source_manifest_sha256'],'source manifest identity')
    for k,v in source_specs().items():check(source_identity(*v)==source[k]['actual_sources'],'frozen source differs: '+k)
    original=read(ROOT/'data/provenance/cao2025-staged-trainer-source-bfee9d8.json')
    check(len(original)==21 and all(sha256(ROOT/'reproduction/cao2025'/k)==v for k,v in original.items()),'original21 changed')
    check(sha256(ROOT/'reproduction/cao2025/formal-protocol.md')=='682fb774e09b9be7f68d09f8251dac6d4b99af3290f02660951d398f9e00eace','original protocol changed')
    pair_path=ROOT/'data/runs'/f'{args.prefix}-pair-audit.json';pair=read(pair_path)
    steps=6 if cycle['engineering'] else 3340
    check(pair['state']=='passed' and pair['steps']==steps and pair['parent_sha256']==PARENT_SHA
          and sha256(pair_path)==cycle['paired_audit_sha256'],'complete paired audit')
    artifacts=[cycle_path,source_path,pair_path]
    for arm,item in pair['arms'].items():
        manifest,audit_path=Path(item['manifest_path']),Path(item['audit_path']);run,audit=read(manifest),read(audit_path)
        check(sha256(manifest)==item['manifest_sha256']==audit['manifest_sha256'] and sha256(audit_path)==item['audit_sha256'],'training/audit identity')
        check(run['state']=='finished' and audit['state']=='passed' and run['arm']==audit['arm']==arm
              and audit['all_optimizer_update_counts']==steps and audit['inactive_states_identical']==519,'training scope')
        check(sha256(run['checkpoint_path'])==run['checkpoint_sha256']==item['checkpoint_sha256']==audit['checkpoint_sha256'],'sole final checkpoint')
        check(sha256(run['full_initialization_path'])==run['full_initialization_sha256']==pair['full_initialization_sha256'],'same exact initial snapshot')
        raw=Path(run['output_dir'])/'training.jsonl';sealed=audit_path.with_suffix('.records')/'training.jsonl'
        check(sha256(raw)==sha256(sealed)==item['raw_snapshot_sha256']==audit['raw_snapshot_sha256'],'complete retained raw records')
        artifacts.extend([manifest,audit_path,sealed])
    for cmd in cycle['completed_commands']:
        path=ROOT/'logs'/f"{args.prefix}-{cmd['label']}.log";check(sha256(path)==cmd['log_sha256'],'retained command log');artifacts.append(path)
    fold=read(ROOT/'data/internal-tuning-fold-001.json');ids=fold['folds']['geocomm_tune_holdout']['ids'];check(len(ids)==372,'ordered holdout scope')
    check(len(cycle['evaluations'])==(0 if cycle['engineering'] else 4),'predetermined endpoint count')
    for rid,item in cycle['evaluations'].items():
        for name,value in item['artifacts_sha256'].items():
            path=ROOT/name;check(sha256(path)==value,'endpoint artifact identity: '+name);artifacts.append(path)
        run=read(ROOT/'data/runs'/f'{rid}.json');AP=read(ROOT/'data/runs'/f'{rid}-AP-audit.json')
        feature=read(ROOT/'data/runs'/f'{rid}-feature-audit.json');boundary=read(ROOT/'data/runs'/f'{rid}-boundary-report.json')
        check(run['state']==boundary['state']=='finished' and AP['state']==feature['state']=='passed','endpoint completion')
        check(AP['frames']==feature['frames']==boundary['frames']==372 and feature['readonly_states']==535
              and len(boundary['final_state_hashes'])==535 and boundary['final_state_hashes']==boundary['checkpoint_loading']['state_hashes'],'full372/535 readonly')
        check(feature['executed_calls']==boundary['calls']==dict(frames=372,student=744,codec=372,channel=372,build_cost=372,forbidden=0),'received-only native inference')
        check(AP['communication']['channel']==[item['channel']] and AP['communication']['complex_uses']==[62400],'actual fixed air condition')
        check(run['mode']=='test' and run['seed']==17 and run['dataset']['split']=='geocomm_tune_holdout' and run['dataset']['count']==372,'fixed test scope')
        check(run['run_checkpoint_sha256']==feature['checkpoint_sha256']==boundary['checkpoint_sha256']==item['checkpoint_sha256'],'fixed final weight')
        records=ROOT/'data/runs'/f'{rid}-features.jsonl';check([json.loads(line)['frame_id'] for line in records.read_text().splitlines()]==ids,'full ordered feature records')
        metrics=Path(run['metrics']['path']);check(sha256(metrics)==run['metrics']['sha256']==AP['metrics_sha256'],'native metrics identity')
        directory=metrics.parent;check(sha256(directory/'result.pkl')==AP['result_pickle_sha256'],'native resultpickle')
        check(set(AP['files'])==set(ids),'native prediction file coverage');data=Path(run['dataset']['root'])
        for frame in ids:
            file=AP['files'][frame]
            for path,key in [(directory/'final_result/data'/f'{frame}.txt','prediction_sha256'),
              (data/'training/label_2'/f'{frame}.txt','label_sha256'),(data/'training/calib'/f'{frame}.txt','calibration_sha256')]:
                check(sha256(path)==file[key],'fresh native AP file identity: '+frame+' '+key)
    result=dict(state='closed_all_audits_passed',prefix=args.prefix,checked_at_unix=time.time(),engineering=cycle['engineering'],
      training_updates_per_arm=steps,paired_audit_sha256=sha256(pair_path),source_sha256={k:v['actual_sources']['sha256'] for k,v in source.items()},
      original21_unchanged=True,artifacts_sha256={str(path.relative_to(ROOT)):sha256(path) for path in artifacts},
      evaluations=cycle['evaluations'],primary_AWGN10_Moderate_treatment_minus_control_pp=cycle.get('primary_AWGN10_Moderate_treatment_minus_control_pp'),
      limitations=cycle['limitations'])
    closure.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:result[k] for k in ('state','prefix','engineering','training_updates_per_arm')}))

if __name__=='__main__':main()
