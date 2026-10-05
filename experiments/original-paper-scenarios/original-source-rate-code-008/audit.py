"""Whole saved-state/order/counter audit; original independent schedule routines."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'reproduction/cao2025'))
from source_loss import SemanticRateVariant,FACTORS
from wireless import WirelessVariant
from audit_training import policy,component,validate_states,optimizer_audit
EPOCHS={1:12,2:10,3:6,4:10}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--terminal',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--native-audit',type=Path);p.add_argument('--predecessor-checkpoint',type=Path);a=p.parse_args();assert not a.output.exists()
 directory=a.directory.resolve();r=read(directory/'report.json');assert r['nominal_rate'] in [10,50] and r['source_linear_factors']==list(FACTORS[r['nominal_rate']]);lock=read(HERE/'inputs.json');assert r['input_sha256']==sha(HERE/'inputs.json') and r['sources_before']==r['sources_after']==lock['files']
 assert all(sha(ROOT/k)==v for k,v in lock['files'].items());assert r['state']=='finished_complete_public_semantic_stage_pending_actual_terminal_and_audit'
 terminal=read(a.terminal);assert terminal['job_id']==r['job_id'] and terminal['actual_job_and_srun_step_completed_exit0']
 assert {r['job_id'],r['job_id']+'.0'}<=set(terminal['rows'])
 for key in [r['job_id'],r['job_id']+'.0']:assert terminal['rows'][key]['State']=='COMPLETED' and terminal['rows'][key]['ExitCode']=='0:0'
 rows=[json.loads(line) for line in (directory/'updates.jsonl').read_text().splitlines()];assert sha(directory/'updates.jsonl')==r['records_sha256']
 ids=read(ROOT/'data/provenance/cost-field-full-dataset-manifest-004.json')['splits']['train']['ids'];ROI={v['frame_id']:v for v in [json.loads(line) for line in (ROOT/'data/engineering/original-full-train-ROI-002/records.jsonl').read_text().splitlines()]}
 initial=torch.load(directory/'initial.pt',map_location='cpu',weights_only=True)['model_state'];assert sha(directory/'initial.pt')==r['initial_sha256'];torch.set_num_threads(2);torch.manual_seed(17);model=WirelessVariant(SemanticRateVariant(ROOT/lock['spynet_path'],r['nominal_rate']));names=list(dict(model.named_parameters()));assert len(names)==718 and len(initial)==768 and sum(x.numel() for x in model.parameters())=={10:12968818,50:13501758}[r['nominal_rate']]
 if r['stage']==1:assert all(torch.equal(initial[k],v) for k,v in model.state_dict().items())
 else:
  assert a.predecessor_checkpoint is not None and sha(a.predecessor_checkpoint)==r['predecessor_checkpoint_sha256'];previous=torch.load(a.predecessor_checkpoint,map_location='cpu',weights_only=True)['model_state'];assert initial.keys()==previous.keys() and all(torch.equal(initial[k],v) for k,v in previous.items());del previous
 del model;count={k:0 for k in names};updates=0;checks=[];stage=r['stage'];epochs=1 if r['scope']=='engineering' else EPOCHS[stage];assert len(r['completed_epochs'])==epochs and len(rows)==r['update_count']==(3 if r['scope']=='engineering' else 3712*epochs)
 for summary in r['completed_epochs']:
  epoch=summary['epoch'];rates,loss=policy(stage,epoch);active={k for k in names if component(k) in rates};assert summary['phase']==dict(stage=stage,epoch=epoch,rates=rates,loss=loss)
  chunk=rows if r['scope']=='engineering' else rows[(epoch-1)*3712:epoch*3712];order=ids[:3] if r['scope']=='engineering' else np.random.default_rng(17+1000*stage+epoch).permutation(ids).tolist();assert [v['frame_id'] for v in chunk]==order
  for index,v in enumerate(chunk):
   updates+=1;native=ROI[v['frame_id']];assert v['step']==updates and v['epoch']==epoch and v['index']==index and v['shape']==[1,3,*native['views'][0]['shape']] and v['box_counts']==[len(x['boxes']) for x in native['views']]
   assert v['source_png_sha256']=={f"data/kitti/training/{x['camera']}/{v['frame_id']}.png":x['image_sha256'] for x in native['views']}
   assert v['loss_type']==loss and v['gradient_parameter_tensors']==len(active) and v['all_gradients_parameters_moments_finite'] and np.isfinite(v['loss']) and v['loss']>=0 and np.isfinite(v['preclip_grad_norm']) and v['preclip_grad_norm']>=0
   for k in active:count[k]+=1
  path=directory/Path(summary['path']).name;assert sha(path)==summary['sha256'];saved=torch.load(path,map_location='cpu',weights_only=True)
  assert saved['nominal_rate']==r['nominal_rate'] and saved['stage']==stage and saved['epoch']==epoch and saved['updates']==summary['updates']==updates and saved['parameter_update_counts']==summary['parameter_update_counts']==count
  frozen,changed=validate_states(initial,saved['model_state'],active,stage);states=optimizer_audit(saved,names,count,stage)
  assert summary['frozen_states_identical'] and summary['frozen_state_tensors']==frozen and summary['samples']==len(chunk)
  checks.append(dict(epoch=epoch,samples=len(chunk),updates=updates,frozen_states_identical=frozen,changed_allowed_states=changed,optimizer_states=states,checkpoint_sha256=sha(path)));del saved
 result=dict(state='passed_complete_original_source_rate_semantic_stage008',nominal_rate=r['nominal_rate'],scope=r['scope'],stage=stage,mode='local' if a.native_audit else 'native',updates=updates,epochs=checks,report_sha256=sha(directory/'report.json'),records_sha256=sha(directory/'updates.jsonl'),input_sha256=sha(HERE/'inputs.json'),verifier_sha256=sha(__file__),actual_terminal_closed=True,terminal_sha256=sha(a.terminal),torch_version=torch.__version__,numpy_version=np.__version__,limitation='Complete source/order and saved-state/Adam-counter/moment/frozen-state proof; per-update raw forward/backward and Adam numerical transitions not independently recomputed.')
 if a.native_audit:
  native=read(a.native_audit);assert native['state']==result['state'] and all(native[k]==result[k] for k in ['nominal_rate','scope','stage','updates','epochs','report_sha256','records_sha256','input_sha256','verifier_sha256','terminal_sha256']);result['native_audit_sha256']=sha(a.native_audit)
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
