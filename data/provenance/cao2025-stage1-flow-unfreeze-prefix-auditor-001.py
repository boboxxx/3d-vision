"""Independent sealed completed epoch3 audit, outside both frozen source trees."""
import hashlib,json,math,time
from pathlib import Path
import torch
root=Path('/home/sheng/paper6');rid='cao2025-full-native-seed17-001-stage1'
output=root/'data/runs/cao2025-stage1-flow-unfreeze-prefix-audit-001.json'
if output.exists(): raise ValueError('preserve previous audit')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
run=json.loads((root/'data/runs'/f'{rid}.json').read_text())
assert sha(run['initialization_path'])==run['initialization_sha256']
initial=torch.load(run['initialization_path'],map_location='cpu',weights_only=False)['model_state'];assert len(initial)==768
completed=next(e for e in run['epochs'] if e['epoch']==3);assert sha(completed['checkpoint_path'])==completed['checkpoint_sha256']
saved=torch.load(completed['checkpoint_path'],map_location='cpu',weights_only=False)
assert saved['samples']==saved['optimizer_updates']==10020 and saved['epoch']==3 and saved['stage']==1
states=saved['model_state'];opt=saved['optimizer_state'];assert len(states)==768 and states.keys()==initial.keys()
lines=(Path(run['output_dir'])/'training.jsonl').read_text().splitlines()[:10020];assert len(lines)==10020
prefix=output.with_suffix('.records.jsonl');prefix.write_text('\n'.join(lines)+'\n')
rows=[json.loads(line) for line in lines]
ids=set(json.loads((root/'data/internal-tuning-fold-001.json').read_text())['folds']['geocomm_tune_train']['ids']);assert len(ids)==3340
for epoch in (1,2,3):
 part=rows[(epoch-1)*3340:epoch*3340];assert len(part)==3340 and set(r['frame_id'] for r in part)==ids
 for index,r in enumerate(part,1+(epoch-1)*3340):
  assert r['sample']==r['optimizer_updates']==index and r['epoch']==epoch
  assert r['gradient_parameter_tensors']==(562 if epoch<=2 else 622)
  assert r['updated'] and r['erasure'] is None and r['accounting'] is None and r['snr_db'] is None
  assert r['loss_type']=='global_charbonnier'
  assert all(math.isfinite(r[k]) and r[k]>=0 for k in ('loss','preclip_grad_norm'))
mapping={};active=set();flow=set();rates={'global_encoder':.0002,'global_decoder':.0002,'fusions':.0001,'flow':.000025}
assert len(opt['param_groups'])==4
for group in opt['param_groups']:
 assert group['component'] in rates and group['lr']==rates[group['component']] and group['weight_decay']==0.
 assert group['betas']==(.9,.999) and group['eps']==1e-8 and len(group['params'])==len(group['parameter_names'])
 for key,name in zip(group['params'],group['parameter_names']):
  assert key not in mapping and name not in active;mapping[key]=name;active.add(name)
  if group['component']=='flow': flow.add(name)
assert len(active)==622 and len(flow)==60 and set(opt['state'])==set(mapping)
frozen=0;changed=[];flow_changed=[]
for name,value in states.items():
 assert value.shape==initial[name].shape and value.dtype==initial[name].dtype and torch.isfinite(value).all()
 if name not in active: assert torch.equal(value,initial[name]);frozen+=1
 elif not torch.equal(value,initial[name]):
  changed.append(name)
  if name in flow: flow_changed.append(name)
assert frozen==146
counts=saved['parameter_update_counts'];assert len(counts)==718
assert all(count==(3340 if name in flow else 10020 if name in active else 0) for name,count in counts.items())
for key,state in opt['state'].items():
 name=mapping[key];assert float(state['step'])==(3340 if name in flow else 10020)
 for field in ('exp_avg','exp_avg_sq'): assert state[field].shape==states[name].shape and torch.isfinite(state[field]).all()
 assert (state['exp_avg_sq']>=0).all()
original=json.loads((root/'data/provenance/cao2025-staged-trainer-source-bfee9d8.json').read_text())
assert len(original)==21 and all(sha(root/'reproduction/cao2025'/name)==identity for name,identity in original.items())
record=dict(state='passed_prefix_only',scope='completed stage1epoch3 flow-unfreeze checkpoint/actualAdam/contiguous first3epochs audit; whole12-epoch stage still incomplete',
 checked_at_unix=time.time(),epochs=3,sealed_samples=10020,complete_states=768,inactive_states_identical=146,
 actual_Adam_parameter_states=622,nonflow562_Adam_steps=10020,flow60_Adam_steps=3340,
 changed_active_states=len(changed),changed_flow_states=len(flow_changed),changed_flow_names=flow_changed,
 original21sources_unchanged=True,raw_prefix_sha256=sha(prefix),checkpoint_sha256=completed['checkpoint_sha256'],
 initialization_sha256=run['initialization_sha256'],source_sha256=sha(__file__),
 limitations='live parent manifest; saved epoch3 and sealed prefix only, not full-stage/holdout/AP or formal baseline completion')
output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
