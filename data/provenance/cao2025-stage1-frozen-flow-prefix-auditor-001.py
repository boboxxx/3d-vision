import hashlib,json,math,time
from pathlib import Path
import torch
root=Path('/home/sheng/paper6');rid='cao2025-full-native-seed17-001-stage1'
run=json.loads((root/'data/runs'/f'{rid}.json').read_text());initial=torch.load(run['initialization_path'],map_location='cpu',weights_only=False)['model_state']
output=root/'data/runs/cao2025-stage1-frozen-flow-prefix-audit-001.json'
if output.exists(): raise ValueError('preserve prior audit')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(run['initialization_path'])==run['initialization_sha256'] and len(initial)==768
completed=run['epochs'][:2];assert len(completed)==2 and [e['epoch'] for e in completed]==[1,2]
lines=(Path(run['output_dir'])/'training.jsonl').read_text().splitlines()[:6680];assert len(lines)==6680
prefix=output.with_suffix('.records.jsonl');prefix.write_text('\n'.join(lines)+'\n')
rows=[json.loads(line) for line in lines];train_ids=set(json.loads((root/'data/internal-tuning-fold-001.json').read_text())['folds']['geocomm_tune_train']['ids'])
for e in (1,2):
 part=rows[(e-1)*3340:e*3340];assert len(part)==3340 and set(r['frame_id'] for r in part)==train_ids
 for i,r in enumerate(part,1+(e-1)*3340):
  assert r['sample']==r['optimizer_updates']==i and r['epoch']==e and r['gradient_parameter_tensors']==562
  assert r['updated'] and r['erasure'] is None and r['accounting'] is None and r['snr_db'] is None
  assert r['loss_type']=='global_charbonnier' and all(math.isfinite(r[k]) and r[k]>=0 for k in ['loss','preclip_grad_norm'])
result=[]
for e in completed:
 saved=torch.load(e['checkpoint_path'],map_location='cpu',weights_only=False);states=saved['model_state'];opt=saved['optimizer_state'];steps=e['epoch']*3340
 assert sha(e['checkpoint_path'])==e['checkpoint_sha256'] and len(states)==768 and states.keys()==initial.keys()
 assert saved['samples']==saved['optimizer_updates']==steps and saved['epoch']==e['epoch'] and saved['stage']==1
 mapping={};active=set();rates={'global_encoder':.0002,'global_decoder':.0002,'fusions':.0001,'flow':.000025}
 assert len(opt['param_groups'])==4
 for group in opt['param_groups']:
  assert group['component'] in rates and group['lr']==rates[group['component']] and group['weight_decay']==0. and group['betas']==(.9,.999) and group['eps']==1e-8
  for key,name in zip(group['params'],group['parameter_names']):
   assert key not in mapping;mapping[key]=name
   if group['component']!='flow': active.add(name)
 assert len(active)==562 and set(opt['state'])=={key for key,name in mapping.items() if name in active}
 frozen=changed=0
 for name,value in states.items():
  assert value.shape==initial[name].shape and value.dtype==initial[name].dtype and torch.isfinite(value).all()
  if name not in active: assert torch.equal(value,initial[name]);frozen+=1
  elif not torch.equal(value,initial[name]): changed+=1
 assert frozen==206
 counts=saved['parameter_update_counts'];assert len(counts)==718 and all(count== (steps if name in active else 0) for name,count in counts.items())
 for key,state in opt['state'].items():
  name=mapping[key];assert float(state['step'])==steps
  for field in ['exp_avg','exp_avg_sq']: assert state[field].shape==states[name].shape and torch.isfinite(state[field]).all()
  assert (state['exp_avg_sq']>=0).all()
 result.append(dict(epoch=e['epoch'],samples=3340,actual_Adam_steps_all562=steps,frozen_states_identical=206,changed_active_states=changed,checkpoint_sha256=e['checkpoint_sha256']))
 del saved,states,opt
record=dict(state='passed_prefix_only',scope='independent completed stage1epochs1-2 frozen-flow checkpoint/Adam/scalar prefix audit; live whole12-epoch stage incomplete',checked_at_unix=time.time(),epochs=result,raw_prefix_sha256=sha(prefix),initialization_sha256=run['initialization_sha256'],source_sha256=sha(__file__),limitations='no completed-stage/holdout/AP claim; parent manifests remain live; saved raw prefix is sealed independently')
output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
