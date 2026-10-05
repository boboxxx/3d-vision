"""Whole saved-wire/target/gradient and actual terminal native engineering closure."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DIR=ROOT/'data/engineering/cost-field-GPU-engineering-001'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 r=json.loads((DIR/'report.json').read_text());assert r['state']=='passed_native_received_only_cost_field_detection_loss_codec_backward_pending_actual_terminal_and_local_proof'
 assert str(r['job_id'])=='11424889'
 queue=subprocess.check_output(['squeue','-h','-j','11424889','-o','%i %T %N'],text=True)
 accounting=subprocess.check_output(['sacct','-j','11424889','--noheader','--parsable2','--format=JobID,State,ExitCode'],text=True)
 assert not queue.strip() and '11424889|COMPLETED|0:0' in accounting and '11424889.0|COMPLETED|0:0' in accounting
 lock=json.loads((ROOT/'experiments/cost-field/GPU-inputs-001.json').read_text())
 assert r['input_sha256']==sha(ROOT/'experiments/cost-field/GPU-inputs-001.json')
 assert r['sources_before']==r['sources_after']==lock['frozen_inputs']
 assert all(sha(ROOT/p)==v for p,v in lock['frozen_inputs'].items())
 assert r['base_freeze_before']==r['base_freeze_after']==lock['base_freeze']
 assert r['loaded_states']==r['final_states'] and len(r['loaded_states'])==484
 assert r['codec_initial_states']==r['codec_final_states'] and r['codec_parameters_per_arm']==1650
 assert sha(DIR/'codec-initial.pt')==r['codec_initial_checkpoint_sha256']
 assert len(r['frames'])==14;values=0;uses=0;artifacts={str((DIR/'report.json').relative_to(ROOT)):sha(DIR/'report.json'),str((DIR/'codec-initial.pt').relative_to(ROOT)):sha(DIR/'codec-initial.pt')}
 for row in r['frames']:
  assert row['cut_calls']==1 and row['forbidden_calls']==0 and row['all_parameter_gradients_absent']
  assert row['readonly_states']==r['loaded_states'] and row['codec_states']==r['codec_initial_states']
  p=ROOT/row['artifact'];assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes'];artifacts[row['artifact']]=sha(p)
  with np.load(p,allow_pickle=False) as f:
   assert set(f.files)==set(row['arrays'])
   for k in f.files:
    a=f[k];assert np.isfinite(a).all()
    assert dict(dtype=str(a.dtype),shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest())==row['arrays'][k];values+=a.size
   assert int((f['assigned_box_cls_labels']>0).sum())==row['positive_anchors']>0
   if row['arm']!='native_identity':
    tx=f['wire_tx'];rx=f['wire_received'];assert tx.shape==rx.shape==(1,29640,2)
    energy=float(np.square(tx.astype(np.float64)).sum());assert abs(energy-row['energy'])<1e-8 and abs(energy-29640)<29640e-6
    assert row['complex_uses']==29640 and row['received_only_decode_identical_after_private_clear']
    if row['channel']=='identity':assert np.array_equal(tx,rx)
    for k in row['codec_gradient_nonzero']:assert int(np.count_nonzero(f[k]))==row['codec_gradient_nonzero'][k]
    uses+=29640
 for p,item in lock['sensor_inputs'].items():assert sha(ROOT/p)==item['sha256'] and (ROOT/p).stat().st_size==item['bytes']
 for p in (DIR/'inputs').iterdir():artifacts[str(p.relative_to(ROOT))]=sha(p)
 result=dict(state='closed_actual_native_cost_field14_case_task_gate',actual_terminal=True,job_id='11424889',checked_unix=time.time(),sacct=accounting,squeue=queue,
  report_sha256=sha(DIR/'report.json'),complete_saved_array_values=int(values),total_complex_uses=uses,artifacts_sha256=artifacts,verifier_sha256=sha(__file__),
  original484_states_and_all_codec_states_unchanged=True,limitation='Untrained received-only task/backward engineering; normalized RBF geometry cancellation remains, no AP/training or geometry benefit')
 target=DIR/'terminal.json';assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('state','complete_saved_array_values','total_complex_uses')}))
if __name__=='__main__':main()
