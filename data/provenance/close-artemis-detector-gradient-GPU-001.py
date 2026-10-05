"""Read-only actual-terminal and whole-array closure for native supervised backward."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DIR=ROOT/'data/engineering/artemis-detector-gradient-GPU-001'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 r=json.loads((DIR/'report.json').read_text())
 assert r['state']=='passed_native_detection_loss_RGB_backward_pending_actual_terminal_and_local_proof'
 job=str(r['job_id']);assert job=='11424856'
 queue=subprocess.check_output(['squeue','-h','-j',job,'-o','%i %T %N'],text=True)
 accounting=subprocess.check_output(['sacct','-j',job,'--noheader','--parsable2','--format=JobID,State,ExitCode'],text=True)
 assert not queue.strip()
 entries={line.split('|')[0]:line.split('|')[1:3] for line in accounting.splitlines() if line.strip()}
 assert entries[job]==entries[job+'.0']==['COMPLETED','0:0']
 lock=json.loads((ROOT/'experiments/artemis-detector-framework/gradient-inputs-001.json').read_text())
 assert sha(ROOT/'experiments/artemis-detector-framework/gradient-inputs-001.json')==r['input_sha256']
 assert r['sources_before']==r['sources_after']==lock['frozen_inputs']
 assert all(sha(ROOT/p)==digest for p,digest in lock['frozen_inputs'].items())
 assert r['base_freeze_before']==r['base_freeze_after']==lock['base_freeze']
 assert r['all_parameters_frozen'] and r['all_parameter_gradients_absent']
 assert r['loaded_states']==r['final_states'] and len(r['loaded_states'])==484
 artifacts={str((DIR/'report.json').relative_to(ROOT)):sha(DIR/'report.json')};values=0
 for row in r['frames']:
  assert row['readonly_states']==r['loaded_states'] and row['all_parameter_gradients_absent'] and row['forbidden_calls']==0
  p=ROOT/row['artifact'];assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes'];artifacts[row['artifact']]=sha(p)
  with np.load(p,allow_pickle=False) as a:
   assert set(a.files)==set(row['arrays'])
   for k in a.files:
    v=a[k];assert np.isfinite(v).all()
    assert dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(v.tobytes()).hexdigest())==row['arrays'][k]
    values+=v.size
   assert int((a['assigned_box_cls_labels']>0).sum())==row['positive_anchors']>0
   for view in ('left','right'):
    assert int(np.count_nonzero(a[view+'_gradient']))==row['gradient_nonzero'][view]>0
 for p,item in lock['sensor_inputs'].items():
  assert sha(ROOT/p)==item['sha256'] and (ROOT/p).stat().st_size==item['bytes']
 for p in sorted((DIR/'inputs').iterdir()):artifacts[str(p.relative_to(ROOT))]=sha(p)
 result=dict(state='closed_actual_native_detection_loss_RGB_backward_terminal',actual_terminal=True,job_id=job,
  checked_unix=time.time(),verifier_sha256=sha(__file__),sacct=accounting,squeue=queue,
  report_sha256=sha(DIR/'report.json'),artifacts_sha256=artifacts,complete_saved_array_values=int(values),
  full484_states_unchanged=True,source_inputs_unchanged=True,base_runtime_unchanged=True,
  limitation='Actual frozen-detector input autograd only; no numerical gradient oracle, optimizer, sparse GPU teacher, codec training or AP claim')
 target=DIR/'terminal.json';assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ('state','actual_terminal','complete_saved_array_values')}))
if __name__=='__main__':main()
