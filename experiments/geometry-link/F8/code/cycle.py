"""Sequential prelocked F8 native cycle, all saved-state/pair/four endpoint audits."""
import argparse,json,os,re,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path[:0]=[str(HERE),str(ROOT/'src')]
from geocomm.evidence import sha256,source_identity
from conditions import PARENT_SHA,PROTOCOL_SHA,CONFIG_SHA,check
from train import source_specs

def read(path):return json.loads(Path(path).read_text())

def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--engineering',action='store_true')
    p.add_argument('--engineering-prefix');args=p.parse_args()
    check(re.fullmatch(r'[a-z0-9][a-z0-9-]{2,100}',args.prefix) is not None,'safe unique run prefix')
    if not args.engineering:check(args.engineering_prefix is not None,'audited paired native sanity required before formal')
    if args.engineering_prefix is not None:check(re.fullmatch(r'[a-z0-9][a-z0-9-]{2,100}',args.engineering_prefix) is not None,'safe engineering prefix')
    os.chdir(ROOT);runs=ROOT/'data/runs';logs=ROOT/'logs';runs.mkdir(exist_ok=True);logs.mkdir(exist_ok=True)
    output=runs/(args.prefix+'-cycle.json');check(not output.exists(),'preserve previous cycle')
    for arm in ('codec','joint'):
        check(not (runs/f'{args.prefix}-{arm}.json').exists() and not Path(f'/mnt/d/paper6/runs/{args.prefix}-{arm}').exists(),'preserve arm outputs')
    parent=Path('/mnt/d/paper6/runs/stereo-channel-native-seed17-001-awgn/checkpoint_epoch_1.pth')
    protocol=ROOT/'experiments/geometry-link/F8-matched-encoder-adaptation.md';config=ROOT/'configs/tuning/stereo_task_seed17_epoch1.yaml';fold=ROOT/'data/internal-tuning-fold-001.json'
    check(sha256(parent)==PARENT_SHA and sha256(protocol)==PROTOCOL_SHA and sha256(config)==CONFIG_SHA,'prelocked sole parent/protocol/config')
    from eligibility import verify_eligibility
    verify_eligibility()
    cpu=read(ROOT/'data/engineering/F8-sheng-CPU-003.json')
    check(cpu['state']=='passed' and cpu['tests']==7 and cpu['CUDA_VISIBLE_DEVICES']=='','CPU scope/noise gates required')
    check(cpu['source_sha256']=={str(p.relative_to(ROOT)):sha256(p) for p in HERE.glob('*.py')},'CPU gate source differs')
    specs=source_specs();identities={k:source_identity(*v) for k,v in specs.items()}
    probe=read(ROOT/'data/engineering/original-StereoRCNN-RGB-GPU-endpoint-probe-001.json')
    check(probe['state']=='passed' and probe['detector_states_readonly']==670 and probe['codec_states_readonly']==768,'terminal native detector engineering required')
    check(all(identities[k]==probe['source_identities'][k] for k in ('project','liga','mmdet','stereo_rcnn')),'detector engineering source closure')
    source_path=ROOT/'data/provenance'/f'server-source-manifest-{args.prefix}.json';check(not source_path.exists(),'preserve source manifest')
    original=read(ROOT/'data/provenance/cao2025-staged-trainer-source-bfee9d8.json')
    def frozen():
        check(all(source_identity(*v)==identities[k] for k,v in specs.items()),'frozen source changed')
        check(len(original)==21 and all(sha256(ROOT/'reproduction/cao2025'/k)==v for k,v in original.items()),'original21 changed')
        check(sha256(ROOT/'reproduction/cao2025/formal-protocol.md')=='682fb774e09b9be7f68d09f8251dac6d4b99af3290f02660951d398f9e00eace','original protocol changed')
    frozen()
    if not args.engineering:
        sanity_path=runs/f'{args.engineering_prefix}-pair-audit.json';sanity=read(sanity_path)
        check(sanity['state']=='passed' and sanity['evidence_type']=='engineering_only' and sanity['steps']==6
              and sanity['parent_sha256']==PARENT_SHA,'paired engineering scope')
        sanity_closure=read(ROOT/'data/provenance'/f'{args.engineering_prefix}-closure.json')
        check(sanity_closure['state']=='closed_all_audits_passed' and sanity_closure['engineering']
              and sanity_closure['training_updates_per_arm']==6 and sanity_closure['paired_audit_sha256']==sha256(sanity_path),
              'terminal fully closed paired engineering required')
        saved_source=read(ROOT/'data/provenance'/f'server-source-manifest-{args.engineering_prefix}.json')
        check(all(saved_source[k]['actual_sources']==v for k,v in identities.items()),'engineering/current exact sources')
        for arm in ('codec','joint'):
            item=sanity['arms'][arm];check(sha256(item['manifest_path'])==item['manifest_sha256'] and sha256(item['audit_path'])==item['audit_sha256'],'engineering evidence changed')
            check(read(item['audit_path'])['summary']['empty_GT_frames']>0,'each engineering arm must exercise native emptyGT')
    source_path.write_text(json.dumps({k:dict(git_revision=None,actual_sources=v) for k,v in identities.items()},indent=2)+'\n')
    info=dict(state='starting',prefix=args.prefix,pid=os.getpid(),engineering=args.engineering,started_at_unix=time.time(),
      source_manifest_sha256=sha256(source_path),sources_sha256={k:v['sha256'] for k,v in identities.items()},completed_commands=[],evaluations={})
    def save():output.write_text(json.dumps(info,indent=2,allow_nan=False)+'\n')
    def invoke(label,command):
        frozen();path=logs/f'{args.prefix}-{label}.log'
        with path.open('x') as stream:subprocess.run([sys.executable,*map(str,command)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        frozen();info['completed_commands'].append(dict(label=label,log_sha256=sha256(path)));save()
    def memory_gate(required_GiB):
        # Detached cycle waits without restarting or disturbing any existing job.
        while True:
            frozen();lines=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip().splitlines()
            check(len(lines)==1,'single GPU');used,total=map(lambda v:int(v.strip()),lines[0].split(','))
            if (total-used)*2**20>=required_GiB*2**30:break
            info.update(state='waiting_physical_GPU_margin',required_free_GiB=required_GiB,last_physical_free_MiB=total-used);save();time.sleep(15)
        info['state']='running';save()
    save()
    try:
        for arm in ('codec','joint'):
            rid=f'{args.prefix}-{arm}';manifest=runs/(rid+'.json');checkpoint_dir=Path('/mnt/d/paper6/runs')/rid
            common=['--config',config,'--protocol',protocol,'--fold',fold,'--source-manifest',source_path]
            memory_gate(12)
            invoke(arm+'-train',[HERE/'train.py',*common,'--checkpoint',parent,'--arm',arm,'--output-dir',checkpoint_dir,'--manifest',manifest,*(['--engineering-steps','6'] if args.engineering else [])])
            invoke(arm+'-training-audit',[HERE/'audit.py',*common,'--initialization',parent,'--manifest',manifest,'--output',runs/(rid+'-training-audit.json'),*(['--engineering-sanity'] if args.engineering else [])])
        pair_path=runs/(args.prefix+'-pair-audit.json')
        invoke('pair-audit',[HERE/'pair_audit.py','--codec-manifest',runs/(args.prefix+'-codec.json'),
          '--joint-manifest',runs/(args.prefix+'-joint.json'),'--output',pair_path,*(['--engineering-sanity'] if args.engineering else [])])
        pair=read(pair_path);check(pair['state']=='passed','paired actual data/RNG audit required');info['paired_audit_sha256']=sha256(pair_path)
        if not args.engineering:
            for arm in ('codec','joint'):
                training=read(runs/f'{args.prefix}-{arm}.json');checkpoint=Path(training['checkpoint_path']);checkpoint_sha=training['checkpoint_sha256']
                for channel in ('identity','awgn'):
                    memory_gate(12);erid=f'{args.prefix}-{arm}-test-{channel}';manifest=runs/(erid+'.json');features=runs/(erid+'-features.jsonl');boundary=runs/(erid+'-boundary-report.json')
                    ap=runs/(erid+'-AP-audit.json');feature_audit=runs/(erid+'-feature-audit.json')
                    invoke(arm+'-test-'+channel,['-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=1',ROOT/'scripts/diagnose_stereo_features.py',
                      '--records',features,'--report',boundary,'--checkpoint-sha256',checkpoint_sha,'--expected-channel',channel,
                      'test','--seed','17','--manifest',manifest,'--protocol',protocol,'--cfg_file',ROOT/f'configs/diagnostic/stereo_feature_{channel}_holdout.yaml',
                      '--ckpt',checkpoint,'--batch_size','1','--workers','4','--launcher','pytorch','--save_to_file','--eval_tag',erid])
                    invoke(arm+'-'+channel+'-AP-audit',[ROOT/'scripts/audit_liga_run.py','--manifest',manifest,'--source-manifest',source_path,
                      '--preparation',ROOT/'data/kitti-preparation-001.json','--tuning-fold',fold,'--stereo-feature-link',
                      *(['--identity-codec-diagnostic'] if channel=='identity' else []),'--output',ap])
                    invoke(arm+'-'+channel+'-feature-audit',[ROOT/'scripts/audit_stereo_features.py','--features',features,'--report',boundary,'--manifest',manifest,
                      '--AP-audit',ap,'--fold',fold,'--expected-channel',channel,'--checkpoint-sha256',checkpoint_sha,'--output',feature_audit])
                    aa,fa,br,run=[read(path) for path in (ap,feature_audit,boundary,manifest)]
                    check(aa['state']==fa['state']=='passed' and run['state']==br['state']=='finished','complete endpoint/audits')
                    check(aa['frames']==fa['frames']==br['frames']==372 and fa['readonly_states']==535,'full372/535 endpoint')
                    check(aa['run_manifest_sha256']==fa['run_manifest_sha256']==sha256(manifest)
                      and fa['AP_audit_sha256']==sha256(ap) and fa['boundary_report_sha256']==sha256(boundary)
                      and fa['feature_records_sha256']==sha256(features),'complete endpoint artifact identity')
                    check(run['run_checkpoint_sha256']==fa['checkpoint_sha256']==br['checkpoint_sha256']==checkpoint_sha,'same sole final checkpoint')
                    info['evaluations'][erid]=dict(arm=arm,channel=channel,checkpoint_sha256=checkpoint_sha,
                      artifacts_sha256={str(path.relative_to(ROOT)):sha256(path) for path in (manifest,ap,feature_audit,boundary,features)},
                      Car3D_AP_R40_percent={k:v for k,v in aa['recomputed_metrics'].items() if k.startswith('Car_3d/')});save()
            clean=info['evaluations'][args.prefix+'-codec-test-awgn']['Car3D_AP_R40_percent']['Car_3d/moderate_R40']
            noisy=info['evaluations'][args.prefix+'-joint-test-awgn']['Car3D_AP_R40_percent']['Car_3d/moderate_R40']
            info['primary_AWGN10_Moderate_treatment_minus_control_pp']=noisy-clean
        frozen();info.update(state='finished_all_independent_audits_passed',ended_at_unix=time.time(),
          limitations='Fixed exploratory internal paired encoder optimization scope only; author-pretraining overlap, single seed, no novelty/general superiority/mainval/fading/project-completion claim');save()
        print(json.dumps({k:info[k] for k in ('state','prefix','engineering')}),flush=True)
    except BaseException as exc:info.update(state='failed',exception=repr(exc),ended_at_unix=time.time());save();raise

if __name__=='__main__':main()
