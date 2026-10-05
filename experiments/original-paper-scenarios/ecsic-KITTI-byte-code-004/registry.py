"""Complete six-final-model admission; no public or intermediate weight fallback."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3];LAMBDAS=(.001,.003,.01,.03,.1,.3)
CONFIG='data/provenance/ecsic-cs001-official-config-001.json';CONFIG_SHA='ba02cc1239b755a26ca0eb920c3014908c97c45fa210c447c47e96f7527102f8'
DATASET='data/provenance/cost-field-full-dataset-manifest-004.json';DATASET_SHA='ecf1f0e41de5757bd438b2a73252dc999eed80fa9828270beff342ccf6ea1310'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def states(s):
 return {k:dict(dtype=str(v.numpy().dtype),shape=list(v.shape),sha256=hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest()) for k,v in s.items()}
def require_all():
 expected=sha(ROOT/'data/provenance/audit-ecsic-KITTI-formal-003.py');candidates=[]
 for i,value in enumerate(LAMBDAS):
  tag=f'{value:.3g}';base=f'ecsic-KITTI-lambda{tag}-seed17-002';directory=ROOT/'data/runs'/base;reportpath=directory/'report.json';r=read(reportpath)
  assert r['state']=='finished_all37120_source_RD_updates_pending_actual_terminal_full_audit_and_real_byte_calibration' and r['candidate']==i and r['lambda_RD']==value and r['seed']==17 and r['update_count']==37120 and len(r['completed_epochs'])==10
  proofs={}
  for host in ['native','local']:
   path=ROOT/'data/provenance'/f'{base}-{host}-audit-003.json';proof=read(path)
   assert proof['state']=='passed_whole37120_source_training_records_and_all10_epoch_states' and proof['candidate']==i and proof['checked_updates']==37120 and proof['checked_epoch_checkpoints']==10 and proof['verifier_sha256']==expected
   assert proof['metadata']['report_sha256']==sha(reportpath) and proof['updates_jsonl_sha256']==r['updates_jsonl_sha256']
   terminal=[line.split('|') for line in proof['metadata']['actual_terminal'].splitlines()]
   for job in [str(r['job_id']),str(r['job_id'])+'.0']:assert [job,'COMPLETED','0:0'] in terminal
   proofs[host]=(path,proof)
  for field in ['metadata','updates_jsonl_sha256','ordered_CUDA_rng_identity_ledger_sha256','checkpoint_saved_values','protocol_sha256','count_repair_protocol_sha256','verifier_sha256']:
   assert proofs['native'][1][field]==proofs['local'][1][field]
  last=r['completed_epochs'][-1];assert last['epoch']==10 and last['updates']==37120 and last['states']==r['final_states'];checkpoint=ROOT/last['path'];assert sha(checkpoint)==last['sha256']
  candidates.append((r,reportpath,checkpoint,proofs))
 return candidates

def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();out=a.directory.resolve();assert out.is_relative_to(ROOT) and not out.exists()
 torch.set_num_threads(2);assert sha(ROOT/CONFIG)==CONFIG_SHA and sha(ROOT/DATASET)==DATASET_SHA
 admitted=require_all();out.mkdir(parents=True);entries=[]
 for r,reportpath,checkpointpath,proofs in admitted:
  checkpoint=torch.load(checkpointpath,map_location='cpu',weights_only=True);model=checkpoint['model'];assert checkpoint['epoch']==10 and checkpoint['updates']==37120 and checkpoint['lambda_RD']==r['lambda_RD'] and len(model)==225 and states(model)==r['final_states']
  assert all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in model.values())
  aliases=[k for k in model if k.startswith(tuple(f'E.{i}.' for i in range(6))) and '.layer_right.' in k];assert len(aliases)==9 and all(torch.equal(model[k],model[k.replace('.layer_right.','.layer_left.')]) for k in aliases)
  target=out/f"lambda{r['lambda_RD']:.3g}-epoch10.pt";torch.save(model,target)
  reload=torch.load(target,map_location='cpu',weights_only=True);assert states(reload)==r['final_states'];del reload
  entries.append(dict(candidate=r['candidate'],lambda_RD=r['lambda_RD'],seed=17,epoch=10,updates=37120,model_path=str(target.relative_to(ROOT)),model_sha256=sha(target),model_bytes=target.stat().st_size,states=r['final_states'],state_entries=225,independent_parameter_tensors=216,shared_aliases=aliases,training_report_path=str(reportpath.relative_to(ROOT)),training_report_sha256=sha(reportpath),source_checkpoint_path=str(checkpointpath.relative_to(ROOT)),source_checkpoint_sha256=sha(checkpointpath),formal_proofs={k:dict(path=str(path.relative_to(ROOT)),sha256=sha(path)) for k,(path,proof) in proofs.items()}));del checkpoint,model
 registry=dict(schema='ecsic-final-KITTI-registry-v1',state='closed_all_six_final_KITTI_models_with_native_local_full_training_admission',entries=entries,official_source_audit_sha256=sha(ROOT/'data/provenance/ECSIC-official-source-audit-001.json'),config_sha256=CONFIG_SHA,dataset_manifest_sha256=DATASET_SHA,registrar_sha256=sha(__file__),protocol_sha256=sha(ROOT/'experiments/original-paper-scenarios/ecsic-KITTI-registry-protocol-005.md'),created_unix=time.time(),torch_version=torch.__version__,calibration_or_AP_executed=False)
 (out/'registry.json').write_text(json.dumps(registry,indent=2)+'\n');print(json.dumps(dict(state=registry['state'],models=len(entries),registry_sha256=sha(out/'registry.json'))))
if __name__=='__main__':main()
