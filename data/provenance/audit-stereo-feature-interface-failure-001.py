"""Verify retained zero-update interface failure without rerunning/tuning it."""
import hashlib,json
from pathlib import Path
import torch
root=Path('/home/sheng/paper6');rid='stereo-feature-sanity-001'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=root/'data/runs'/f'{rid}.json';run=json.loads(manifest.read_text())
output=root/'data/runs'/f'{rid}-zero-update-audit.json'
if output.exists(): raise ValueError('preserve prior evidence')
assert run['state']=='failed' and run['optimizer_steps']==0
assert run['exception']=="ValueError('locked stereo/appearance shape agreement required')"
records=Path(run['output_dir'])/'training.jsonl';assert records.exists() and records.read_bytes()==b''
assert not (Path(run['output_dir'])/'checkpoint_sanity.pth').exists()
assert not (Path(run['output_dir'])/'checkpoint_epoch_1.pth').exists()
assert sha(run['full_initialization_path'])==run['full_initialization_sha256']
assert sha('/mnt/d/paper6/runs/liga/home_sheng_paper6_configs_tuning/student_task_seed17_epoch1.student-task-seed17-001/ckpt/checkpoint_epoch_1.pth')==run['initialization_sha256']
predecessor=torch.load('/mnt/d/paper6/runs/liga/home_sheng_paper6_configs_tuning/student_task_seed17_epoch1.student-task-seed17-001/ckpt/checkpoint_epoch_1.pth',map_location='cpu',weights_only=False)['model_state']
initial=torch.load(run['full_initialization_path'],map_location='cpu',weights_only=False)['model_state']
assert len(predecessor)==519 and len(initial)==535 and set(predecessor)<=set(initial)
assert all(initial[k].shape==v.shape and initial[k].dtype==v.dtype and torch.equal(initial[k],v) for k,v in predecessor.items())
assert all(torch.isfinite(v).all() for v in initial.values())
new=set(initial)-set(predecessor);assert len(new)==16 and all(k.startswith('backbone_3d.stereo_feature_link.') for k in new)
assert tuple(initial['backbone_3d.stereo_feature_link.stereo_encoder.3.weight'].shape)==(4,32,1,1)
assert tuple(initial['backbone_3d.stereo_feature_link.stereo_decoder.0.weight'].shape)==(32,4,3,3)
start=json.loads((root/'data/provenance'/f'server-source-manifest-{rid}.json').read_text())
closure=json.loads((root/'data/provenance'/f'server-source-closure-{rid}.json').read_text())
for tree in start: assert start[tree]['actual_sources']==closure[tree]['actual_sources']
result=dict(state='passed_zero_update_failure_only',manifest_sha256=sha(manifest),raw_records_sha256=sha(records),
 saved_full_initialization_sha256=run['full_initialization_sha256'],exact_F2_states_in_initialization=519,
 original_new_codec_states=16,optimizer_updates=0,learned_checkpoint_created=False,all4_pre_failure_source_trees_closed=True,
 auditor_sha256=sha(__file__),limitations='preserved failed interface before loss/backward/update; no AP or training outcome; current corrected F5b code is separate evidence')
output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
