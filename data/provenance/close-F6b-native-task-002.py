"""Terminal F6b full artifact/source closure; does not create performance evidence."""
import json,os,sys,time
from pathlib import Path
ROOT=Path('/home/sheng/paper6');sys.path.insert(0,str(ROOT/'src'))
from geocomm.evidence import source_identity,sha256
RID='stereo-native-task-seed17-002'
def read(path):return json.loads(Path(path).read_text())
def check(value,message):
 if not value:raise RuntimeError(message)
closure=ROOT/'data/provenance'/f'{RID}-closure.json'
check(not closure.exists(),'retain closure')
launch=read(ROOT/'data/provenance'/f'{RID}-launch.json')
try:os.kill(launch['pid'],0)
except ProcessLookupError:pass
else:raise RuntimeError('cycle stilllive; do not release frozen sources')
check(f'F6 clean native3D codec task adaptation finished: {RID}' in (ROOT/'logs'/f'{RID}-cycle.log').read_text(),'no success marker')
source_path=ROOT/'data/provenance'/f'server-source-manifest-{RID}.json';source=read(source_path)
specs={'project':(ROOT,['src','scripts','configs','pyproject.toml']),
 'liga':(ROOT/'third_party/LIGA-Stereo',['liga','configs','tools','setup.py']),
 'mmdet':(ROOT/'third_party/mmdetection_kitti',['mmdet']),
 'stereo_rcnn':(ROOT/'third_party/Stereo-RCNN',['lib','demo.py','test_net.py'])}
actual={name:dict(git_revision=None,actual_sources=source_identity(*spec)) for name,spec in specs.items()}
for name in specs:check(actual[name]['actual_sources']==source[name]['actual_sources'],'source changed: '+name)
original=read(ROOT/'data/provenance/cao2025-staged-trainer-source-bfee9d8.json')
check(len(original)==21 and all(sha256(ROOT/'reproduction/cao2025'/name)==identity for name,identity in original.items()),'original21 changed')
check(sha256(ROOT/'reproduction/cao2025/formal-protocol.md')=='682fb774e09b9be7f68d09f8251dac6d4b99af3290f02660951d398f9e00eace','original protocol changed')
train_path=ROOT/'data/runs'/f'{RID}.json';audit_path=ROOT/'data/runs'/f'{RID}-training-audit.json'
train,audit=read(train_path),read(audit_path)
check(train['state']=='finished' and train['optimizer_steps']==3340 and train['completed_epochs']==1,'training incomplete')
check(audit['state']=='passed' and audit['steps']==audit['all_optimizer_update_counts']==3340 and audit['inactive_states_identical']==519 and audit['trained_parameter_tensors']==16,'full audit incomplete')
check(audit['manifest_sha256']==sha256(train_path) and audit['source_manifest_sha256']==sha256(source_path),'training audit identities differ')
check(train['checkpoint_sha256']==audit['checkpoint_sha256']==sha256(train['checkpoint_path']),'sole final checkpoint differs')
check(train['initialization_sha256']=='9a4b291070df160e6c03e75ed31eeee1dda7f6bfbaa0528a43a9cdf526977553','wrong F5b init')
check(train['full_initialization_sha256']==audit['full_initialization_sha256']==sha256(train['full_initialization_path']),'full initialsnapshot differs')
raw=Path(train['output_dir'])/'training.jsonl';sealed=audit_path.with_suffix('.records')/'training.jsonl'
check(train['training_records_sha256']==audit['raw_snapshot_sha256']==sha256(raw)==sha256(sealed),'full training rows differ')
artifacts=[train_path,audit_path,sealed];evaluations={}
fold=read(ROOT/'data/internal-tuning-fold-001.json');ids=fold['folds']['geocomm_tune_holdout']['ids'];check(len(ids)==372,'fold differs')
for channel in ('identity','awgn'):
 erid=RID+'-'+channel
 paths={name:ROOT/'data/runs'/(erid+suffix) for name,suffix in {
  'run':'.json','AP':'-AP-audit.json','feature':'-feature-audit.json','boundary':'-boundary-report.json','records':'-features.jsonl'}.items()}
 run,AP,feature,boundary=[read(paths[k]) for k in ('run','AP','feature','boundary')]
 check(run['state']==boundary['state']=='finished' and AP['state']==feature['state']=='passed','eval or audit incomplete '+channel)
 check(run['mode']=='test' and run['seed']==17 and run['dataset']['count']==372 and run['dataset']['split']=='geocomm_tune_holdout','eval scope differs')
 check(run['run_checkpoint_sha256']==train['checkpoint_sha256']==feature['checkpoint_sha256']==boundary['checkpoint_sha256'],'eval checkpoint differs')
 check(AP['frames']==feature['frames']==boundary['frames']==372 and len(AP['files'])==372,'native coverage incomplete')
 check(AP['run_manifest_sha256']==feature['run_manifest_sha256']==sha256(paths['run']),'eval manifest identity differs')
 check(AP['source_manifest_sha256']==sha256(source_path),'AP source differs')
 check(feature['AP_audit_sha256']==sha256(paths['AP']) and feature['boundary_report_sha256']==sha256(paths['boundary']),'feature audit identity differs')
 check(feature['feature_records_sha256']==boundary['records_sha256']==sha256(paths['records']),'raw passive feature identity differs')
 check(feature['readonly_states']==535 and len(boundary['final_state_hashes'])==535 and boundary['final_state_hashes']==boundary['checkpoint_loading']['state_hashes'],'full535 readonly evidence differs')
 check(feature['executed_calls']==boundary['calls']==dict(frames=372,student=744,codec=372,channel=372,build_cost=372,forbidden=0),'received-only inference operation coverage differs')
 check(AP['communication']['frames']==372 and AP['communication']['channel']==[channel] and AP['communication']['complex_uses']==[62400],'physical channel scope differs')
 raw_ids=[json.loads(line)['frame_id'] for line in paths['records'].read_text().splitlines()]
 check(raw_ids==ids,'ordered passive features differ')
 metrics_path=Path(run['metrics']['path']);check(sha256(metrics_path)==run['metrics']['sha256']==AP['metrics_sha256'],'metrics changed')
 evaldir=metrics_path.parent;check(sha256(evaldir/'result.pkl')==AP['result_pickle_sha256'],'native resultpickle changed')
 check(set(AP['files'])==set(ids),'AP prediction coverage differs')
 data_root=Path(run['dataset']['root'])
 for frame in ids:
  file=AP['files'][frame]
  for path,key in [(evaldir/'final_result/data'/f'{frame}.txt','prediction_sha256'),
    (data_root/'training/label_2'/f'{frame}.txt','label_sha256'),(data_root/'training/calib'/f'{frame}.txt','calibration_sha256')]:
   check(sha256(path)==file[key],'native AP input changed '+frame+' '+key)
 for tree,key in [('project','project_sources'),('liga','detector_sources'),('mmdet','mmdet_sources')]:
  check(run[key]==source[tree]['actual_sources'],'run source differs '+tree)
 check(boundary['project_sources']==source['project']['actual_sources'],'boundary source differs')
 artifacts.extend(paths.values())
 evaluations[channel]=dict(Car3D_AP_R40_percent={k:v for k,v in AP['recomputed_metrics'].items() if k.startswith('Car_3d/')},feature_summaries=feature['summaries'])
source_closure=ROOT/'data/provenance'/f'server-source-closure-{RID}.json'
check(not source_closure.exists(),'retain existing sourceclosure')
source_closure.write_text(json.dumps(actual,indent=2)+'\n');artifacts.append(source_closure)
record=dict(state='closed_all_audits_passed',run_id=RID,checked_at_unix=time.time(),training_updates=3340,
 complete_model_states=535,frozen_states_identical=519,actual_Adam_parameters=16,actual_Adam_steps=3340,
 checkpoint_sha256=train['checkpoint_sha256'],full_initialization_sha256=train['full_initialization_sha256'],
 all4_source_trees_unchanged={name:actual[name]['actual_sources']['sha256'] for name in actual},original21sources_unchanged=True,
 artifacts_sha256={str(path.relative_to(ROOT)):sha256(path) for path in artifacts},evaluations=evaluations,
 training_summary=audit['summary'],preserved_failure_closure_sha256=sha256(ROOT/'data/provenance/stereo-native-task-seed17-001-failure-closure.json'),
 limitations='Closed fixed F6b clean-native3D exploratory cycle only; internal author-pretraining overlap, no matched-method superiority, mainval, fulloriginal baseline or project-completion claim')
closure.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:record[k] for k in ['state','checkpoint_sha256','training_updates','evaluations']},ensure_ascii=False))
