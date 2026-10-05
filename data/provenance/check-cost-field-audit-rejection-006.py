"""Real first-step acceptance and consistent metadata-tamper rejection."""
import copy,importlib.util,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
path=ROOT/'data/provenance/audit-cost-field-full-006.py'
spec=importlib.util.spec_from_file_location('audit006',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
FIX=ROOT/'data/engineering/cost-field-audit-fixture-006'
row=json.loads((ROOT/'data/provenance/cost-field-first-update-fixture-006.json').read_text())
initial=m.safe((FIX/'codec-initial.pt').read_bytes());checkpoint=m.safe((FIX/'002053-step0-state.pt').read_bytes())
assert m.sha(FIX/'002053-step0-state.pt')==row['optimizer_checkpoint_sha256'] and m.sha(FIX/'002053-step0.npz')==row['sha256']
with np.load(FIX/'002053-step0.npz',allow_pickle=False) as f:arrays={k:f[k] for k in f.files}

def fresh():
 a=m.Audit.__new__(m.Audit);a.initial=initial;a.previous={arm:dict(codec=s,optimizer=dict(state={})) for arm,s in initial.items()}
 a.r=dict(codec_parameters_per_arm=dict(G=1650,P=1650,S=1650,B=1751));a.maxima=dict(Adam_parameter=0.,Adam_moment=0.,baseband=0.,ZF=0.);a.gradvalues=0;a.uses=0;a.noise=[0.,0.,0];a.fading=[0.,0.,0]
 return a

def reject(label,function):
 try:function()
 except AssertionError as error:return dict(case=label,rejected=True,assertion=str(error)[:300])
 raise AssertionError('Corrupt record incorrectly accepted: '+label)

clean=fresh();clean.optimizer(arrays,row,checkpoint);clean.physics(arrays,row)
results=[]
# Parameter and its descriptor are both changed: a plain SHA check would pass.
tampered=copy.deepcopy(checkpoint);name=row['changed_states'][0];tampered['codec'][name].view(-1)[0]+=.01
metadata=copy.deepcopy(row);metadata['codec_states_after']=m.states(tampered['codec']);metadata['changed_states']=[k for k in metadata['codec_states_before'] if metadata['codec_states_before'][k]!=metadata['codec_states_after'][k]]
results.append(reject('parameter_plus_consistent_descriptor',lambda:fresh().optimizer(arrays,metadata,tampered)))
# Codec states and raw task gradients remain unchanged; alter an Adam moment.
tampered_moment=copy.deepcopy(checkpoint);first_id=tampered_moment['optimizer']['param_groups'][0]['params'][0]
tampered_moment['optimizer']['state'][first_id]['exp_avg'].view(-1)[0]+=.1
results.append(reject('optimizer_moment',lambda:fresh().optimizer(arrays,row,tampered_moment)))
# RX and baseband agree with each other; neither agrees with charged TX+noise.
tampered_wire={k:v.copy() for k,v in arrays.items()};tampered_wire['channel_received_baseband'][0,0,0]+=.1;tampered_wire['wire_received'][0,0,0]+=.1
results.append(reject('consistent_RX_baseband',lambda:fresh().physics(tampered_wire,row)))
output=ROOT/'data/provenance/cost-field-audit-rejection-006.json';assert not output.exists()
result=dict(state='passed_real_clean_first_step_and_three_consistent_tamper_rejections',clean_frame=row['frame_id'],results=results,clean_max_errors=clean.maxima,checker_sha256=m.sha(__file__),auditor_sha256=m.sha(path),fixture_sha256={p.name:m.sha(p) for p in FIX.iterdir()},limitation='Engineering rejection tests only; no independent gradient oracle or validation AP.')
output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(state=result['state'],rejected=len(results))))
