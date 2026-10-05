"""Close the terminated F6 attempt without claiming a completed training audit."""
import hashlib,json,os,sys,time
from pathlib import Path
root=Path('/home/sheng/paper6');sys.path.insert(0,str(root/'src'))
from geocomm.evidence import source_identity,sha256
rid='stereo-native-task-seed17-001'
out=root/'data/provenance'/f'{rid}-failure-closure.json'
assert not out.exists()
run_path=root/'data/runs'/f'{rid}.json';run=json.loads(run_path.read_text())
launch_path=root/'data/provenance'/f'{rid}-launch.json';launch=json.loads(launch_path.read_text())
assert run['state']=='failed' and run['optimizer_steps']==5 and run['exception']=="ValueError('native augmented3D GT layout/finite differs')"
try:os.kill(launch['pid'],0)
except ProcessLookupError:pass
else:raise RuntimeError('failed cycle process still alive')
source_path=root/'data/provenance'/f'server-source-manifest-{rid}.json';before=json.loads(source_path.read_text())
closure_path=root/'data/provenance'/f'server-source-closure-{rid}-failed.json';after=json.loads(closure_path.read_text())
specs={'project':(root,['src','scripts','configs','pyproject.toml']),
 'liga':(root/'third_party/LIGA-Stereo',['liga','configs','tools','setup.py']),
 'mmdet':(root/'third_party/mmdetection_kitti',['mmdet']),
 'stereo_rcnn':(root/'third_party/Stereo-RCNN',['lib','demo.py','test_net.py'])}
for name,spec in specs.items():assert before[name]['actual_sources']==after[name]['actual_sources']==source_identity(*spec)
original=json.loads((root/'data/provenance/cao2025-staged-trainer-source-bfee9d8.json').read_text())
assert len(original)==21 and all(sha256(root/'reproduction/cao2025'/name)==value for name,value in original.items())
os.kill(26642,0)
records=Path(run['output_dir'])/'training.jsonl';rows=[json.loads(line) for line in records.read_text().splitlines()]
assert [r['step'] for r in rows]==list(range(1,6)) and all(r['epoch']==1 for r in rows)
assert not (Path(run['output_dir'])/'checkpoint_epoch_1.pth').exists()
assert not any((root/'data/runs'/f'{rid}-{s}.json').exists() for s in ['identity','awgn'])
assert sha256(run['full_initialization_path'])==run['full_initialization_sha256']
paths=[run_path,launch_path,source_path,closure_path,records,root/'logs'/f'{rid}-train.log',root/'logs'/f'{rid}-cycle.log']
result=dict(state='closed_failed_attempt',run_id=rid,optimizer_steps_recorded=5,raw_rows=5,
 no_final_checkpoint=True,no_final_AP_evaluation=True,all4_source_trees_unchanged=True,
 original21_sources_unchanged=True,original_pid26642_alive=True,closed_at_unix=time.time(),
 exception=run['exception'],files={str(path):sha256(path) for path in paths},
 initial535_snapshot_sha256=run['full_initialization_sha256'],
 limitation='Five recorded updates only; process exited before final state/optimizer checkpoint, no full-state or final performance claim. Repaired independent run must use original F5b initialization, never this attempt weights.')
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
