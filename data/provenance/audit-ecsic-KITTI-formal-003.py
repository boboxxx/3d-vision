"""Whole formal source-training exposure/checkpoint audit; not raw-gradient replay."""
import argparse,hashlib,io,json,math,subprocess,sys,tarfile,time,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
LAMBDAS=(.001,.003,.01,.03,.1,.3)
EPS=np.finfo(np.float32).eps
def digest(b):return hashlib.sha256(b).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def states(s):return {k:dict(dtype=str(v.numpy().dtype),shape=list(v.shape),sha256=digest(v.contiguous().numpy().tobytes())) for k,v in s.items()}
def blobs(candidate):
 path=ROOT/'data/runs'/f'ecsic-KITTI-lambda{LAMBDAS[candidate]:.3g}-seed17-002'
 b=(path/'report.json').read_bytes();r=json.loads(b)
 assert r['state']=='finished_all37120_source_RD_updates_pending_actual_terminal_full_audit_and_real_byte_calibration' and r['update_count']==37120 and len(r['completed_epochs'])==10
 job=str(r['job_id']);accounting=subprocess.check_output(['sacct','-j',job,'--format=JobIDRaw,State,ExitCode','-n','-P'],text=True)
 rows=[x.split('|') for x in accounting.splitlines()]
 for raw in (job,job+'.0'):assert [raw,'COMPLETED','0:0'] in rows,accounting
 yield 'metadata.json',json.dumps(dict(candidate=candidate,report_sha256=digest(b),actual_terminal=accounting)).encode()
 yield 'report.json',b
 with (path/'updates.jsonl').open('rb') as f:
  h=hashlib.sha256()
  for index in range(37120):
   line=f.readline();assert line.endswith(b'\n');h.update(line);yield f'row/{index}.json',line
  assert not f.read(1) and h.hexdigest()==r['updates_jsonl_sha256']
 for epoch in r['completed_epochs']:yield epoch['path'],(ROOT/epoch['path']).read_bytes()
 yield 'footer.json',json.dumps(dict(count=37120,updates_jsonl_sha256=h.hexdigest())).encode()
def exported(candidate):
 with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
  for name,b in blobs(candidate):
   info=tarfile.TarInfo(name);info.size=len(b);archive.addfile(info,io.BytesIO(b))
def incoming():
 with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as archive:
  for member in archive:
   assert member.isfile() and 0<=member.size<=512*1024*1024 and not member.name.startswith('/') and '..' not in Path(member.name).parts
   b=archive.extractfile(member).read();assert len(b)==member.size;yield member.name,b
def audit(iterator,output,candidate):
 it=iter(iterator)
 def get(name):
  actual,b=next(it);assert actual==name,(actual,name);return b
 metadata=json.loads(get('metadata.json'));assert metadata['candidate']==candidate
 b=get('report.json');assert digest(b)==metadata['report_sha256'];r=json.loads(b)
 result=dict(state='failed',candidate=candidate,lambda_RD=LAMBDAS[candidate],seed=17,checked_updates=0,checked_epoch_checkpoints=0,checkpoint_saved_values=0,max_loss_algebra_error=0.,full_raw_gradient_replay=False,actual_source_bytes_calibrated=False,AP_measured=False)
 h=hashlib.sha256();rnghash=hashlib.sha256()
 try:
  lockpath=ROOT/'experiments/original-paper-scenarios/ecsic-KITTI-adaptation-code-002/inputs.json';lock=json.loads(lockpath.read_text());assert sha(lockpath)==r['input_sha256']
  assert r['candidate']==candidate and r['lambda_RD']==LAMBDAS[candidate] and r['seed']==17 and int(r['array_task_id'])==candidate
  assert r['sources_before']==r['sources_after']==lock['files'] and r['base_freeze_before']==r['base_freeze_after']==lock['base_freeze']
  assert r['GPU'].find('RTX PRO 6000')>=0 and r['capability']==[12,0] and r['input_barrier'] and r['no_detector_or_GT'] and not r['formal_raw_gradient_replay']
  assert r['parameter_count']==31640742 and r['parameter_tensors']==216 and r['state_tensors']==225
  assert r['peak_reserved_bytes']+2*1024**3<=r['initial_free_bytes']
  publicpath=ROOT/'assets/ecsic-cs001-001/model.pt';assert sha(publicpath)==r['initialization_sha256']=='e66b55e2f888f5d113ea6c6414640c7486280d6e18838ba77c650720a5636052'
  public=torch.load(publicpath,map_location='cpu',weights_only=True);assert states(public)==r['initial_states'] and len(public)==225
  aliases=[k for k in public if k.startswith(('E.0.','E.1.','E.2.','E.3.','E.4.','E.5.')) and '.layer_right.' in k]
  assert len(aliases)==9
  for k in aliases:assert torch.equal(public[k],public[k.replace('.layer_right.','.layer_left.')])
  del public
  manifestpath=ROOT/'data/provenance/cost-field-full-dataset-manifest-004.json';assert sha(manifestpath)==r['manifest_sha256'];manifest=json.loads(manifestpath.read_text())
  ids=manifest['splits']['train']['ids'];assert len(ids)==len(set(ids))==3712 and not set(ids)&set(manifest['splits']['val']['ids'])
  orders=[np.random.default_rng(17+1000*epoch).permutation(ids).tolist() for epoch in range(10)]
  for index in range(37120):
   line=get(f'row/{index}.json');h.update(line);row=json.loads(line);epoch,position=divmod(index,3712);frame=orders[epoch][position]
   assert (row['step'],row['epoch'],row['index'],row['frame_id'],row['lambda_RD'])==(index,epoch+1,position,frame,LAMBDAS[candidate])
   expected={f'data/kitti/training/{view}/{frame}.png':manifest['files'][f'data/kitti/training/{view}/{frame}.png']['sha256'] for view in ('image_2','image_3')}
   assert row['source_png_sha256']==expected
   hw=row['original_hw'];assert len(hw)==2 and all(type(v)==int and 0<v<=4096 for v in hw)
   assert row['padded_hw']==[(v+31)//32*32 for v in hw]
   mse,bpp,loss,norm=(row[k] for k in ('mse_255scale','estimated_bpp_per_view','loss','gradient_norm_before_clip'))
   assert all(math.isfinite(v) and v>=0 for v in (mse,bpp,loss,norm))
   ref=(bpp+LAMBDAS[candidate]*mse)/(1+LAMBDAS[candidate]);error=abs(loss-ref);assert error<=64*EPS*(1+abs(ref))
   result['max_loss_algebra_error']=max(result['max_loss_algebra_error'],error)
   assert row['gradient_tensors']==216 and row['gradient_values']==31640742 and row['all_gradients_and_parameters_and_Adam_moments_finite']
   assert row['optimizer_step_min']==row['optimizer_step_max']==index+1
   raw=bytes.fromhex(row['CUDA_rng_sha256']);assert len(raw)==32;rnghash.update(raw)
   result['checked_updates']+=1
  assert h.hexdigest()==r['updates_jsonl_sha256'] and [(e['epoch'],e['updates']) for e in r['completed_epochs']]==[(e,3712*e) for e in range(1,11)]
  for epoch in r['completed_epochs']:
   b=get(epoch['path']);assert digest(b)==epoch['sha256'];checkpoint=torch.load(io.BytesIO(b),map_location='cpu',weights_only=True)
   assert checkpoint['epoch']==epoch['epoch'] and checkpoint['updates']==epoch['updates'] and checkpoint['lambda_RD']==LAMBDAS[candidate]
   model=checkpoint['model'];assert states(model)==epoch['states'] and set(model)==set(r['initial_states']) and len(model)==225
   assert all(torch.isfinite(v).all() for v in model.values()) and states(model)!=r['initial_states']
   for k in aliases:assert torch.equal(model[k],model[k.replace('.layer_right.','.layer_left.')])
   groups=checkpoint['optimizer']['param_groups'];assert len(groups)==1;g=groups[0]
   assert g['lr']==1e-4 and tuple(g['betas'])==(.9,.999) and g['eps']==1e-8 and g['weight_decay']==0 and not g.get('amsgrad',False)
   optim=checkpoint['optimizer']['state'];assert len(g['params'])==len(optim)==216 and set(optim)==set(g['params'])
   for state in optim.values():
    assert set(state)=={'step','exp_avg','exp_avg_sq'} and float(state['step'])==epoch['updates']
    assert state['exp_avg'].dtype==state['exp_avg_sq'].dtype==torch.float32 and state['exp_avg'].shape==state['exp_avg_sq'].shape
    assert torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all() and (state['exp_avg_sq']>=0).all()
   assert sum(s['exp_avg'].numel() for s in optim.values())==31640742
   result['checkpoint_saved_values']+=sum(v.numel() for v in model.values())+sum(v.numel() for s in optim.values() for v in s.values())
   result['checked_epoch_checkpoints']+=1
   if epoch['epoch']==10:assert states(model)==r['final_states']
   del checkpoint,model,optim
  footer=json.loads(get('footer.json'));assert footer==dict(count=37120,updates_jsonl_sha256=h.hexdigest())
  try:next(it)
  except StopIteration:pass
  else:raise AssertionError('Unexpected suffix')
  result['state']='passed_whole37120_source_training_records_and_all10_epoch_states'
 except BaseException:result['traceback']=traceback.format_exc();raise
 finally:
  result.update(metadata=metadata,updates_jsonl_sha256=h.hexdigest(),ordered_CUDA_rng_identity_ledger_sha256=rnghash.hexdigest(),initial_CUDA_rng_sha256=r.get('initial_CUDA_rng_sha256'),verifier_sha256=sha(__file__),count_repair_protocol_sha256=sha(ROOT/'experiments/original-paper-scenarios/ecsic-KITTI-formal-audit-alias-repair-003.md'),protocol_sha256=sha(ROOT/'experiments/original-paper-scenarios/ecsic-KITTI-adaptation-protocol-002.md'),checked_unix=time.time(),limitation='Full exposure/scalar/native-invariant logs and all saved epoch weights/moments; not independent full RGB/gradient/Adam transition replay. Actual entropy rates, calibration and AP remain required.')
  output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:result[k] for k in ('state','candidate','checked_updates','checked_epoch_checkpoints')}))
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=('native','export','stream'),required=True);p.add_argument('--candidate',type=int,choices=range(6),required=True);p.add_argument('--output',type=Path);a=p.parse_args();torch.set_num_threads(2)
 if a.mode=='export':exported(a.candidate);return
 assert a.output and not a.output.exists();audit(incoming() if a.mode=='stream' else blobs(a.candidate),a.output,a.candidate)
if __name__=='__main__':main()
