"""Exact actual sm120 replay of every native FP32 depth-regression value."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'data/engineering/artemis-full-detector-GPU-001'
OUT=ROOT/'data/engineering/artemis-depth-FP32-replay-001'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not OUT.exists();OUT.mkdir(parents=True)
 r=dict(state='starting',job_id=os.environ['SLURM_JOB_ID'],started_unix=time.time(),frames=[],code_sha256=sha(__file__))
 def save():(OUT/'report.json').write_text(json.dumps(r,indent=2)+'\n')
 save()
 lock=json.loads((ROOT/'experiments/artemis-detector-framework/depth-replay-inputs-001.json').read_text())
 try:
  torch.set_num_threads(2)
  r['input_sha256']=sha(ROOT/'experiments/artemis-detector-framework/depth-replay-inputs-001.json')
  assert all(sha(ROOT/p)==digest for p,digest in lock['frozen_inputs'].items())
  assert torch.__version__=='2.7.1+cu128' and torch.cuda.device_count()==1 and torch.cuda.get_device_capability(0)==(12,0)
  r['GPU']=torch.cuda.get_device_name();assert 'RTX PRO 6000' in r['GPU']
  r['capability']=[12,0];r['torch_version']=torch.__version__
  source_report=json.loads((SOURCE/'report.json').read_text());closure=json.loads((SOURCE/'terminal.json').read_text())
  assert closure['actual_terminal'] and closure['report_sha256']==sha(SOURCE/'report.json')
  torch.cuda.reset_peak_memory_stats()
  for row in source_report['frames']:
   p=ROOT/row['artifact'];assert sha(p)==row['sha256']
   with np.load(p,allow_pickle=False) as f:logits=f['depth_logits_low'];axis=f['depth_samples'];native=f['depth_pred']
   for key,a in [('depth_logits_low',logits),('depth_samples',axis),('depth_pred',native)]:
    assert hashlib.sha256(a.tobytes()).hexdigest()==row['arrays'][key]['sha256']
   with torch.no_grad():
    x=torch.from_numpy(logits[:,0]).cuda();d=torch.from_numpy(axis).cuda()
    probability=torch.softmax(x,dim=1);value=torch.sum(probability*d[None,:,None,None],dim=1)
    replay=value.cpu().numpy()
   torch.cuda.synchronize();assert np.array_equal(native,replay)
   target=OUT/(row['frame_id']+'.npz');np.savez_compressed(target,depth_FP32_replay=replay)
   r['frames'].append(dict(frame=row['frame_id'],all_values=int(native.size),all_values_bit_equal=True,
    source_sha256=row['sha256'],artifact=str(target.relative_to(ROOT)),sha256=sha(target),
    replay_array_sha256=hashlib.sha256(replay.tobytes()).hexdigest(),dtype=str(replay.dtype),shape=list(replay.shape)))
   save();del logits,axis,native,x,d,probability,value,replay
  r.update(state='passed_every_native_FP32_depth_output_exact_replay_pending_terminal',peak_reserved_bytes=torch.cuda.max_memory_reserved(),
   original_FP64_3e5_criterion_passed=False,detector_constructed=False,optimizer_used=False)
 except BaseException:r['state']='failed';r['traceback']=traceback.format_exc();raise
 finally:
  r['frozen_inputs_unchanged']=all(sha(ROOT/p)==digest for p,digest in lock['frozen_inputs'].items())
  e=os.environ.copy();e.pop('PYTHONPATH',None)
  r['base_freeze_after']=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True,env=e)
  r['base_runtime_unchanged']=r['base_freeze_after']==lock['base_freeze']
  r['ended_unix']=time.time();save();assert r['frozen_inputs_unchanged'] and r['base_runtime_unchanged']
if __name__=='__main__':main()
