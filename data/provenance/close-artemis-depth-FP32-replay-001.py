"""Actual terminal native closure and full independent transferred replay equality."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DIR=ROOT/'data/engineering/artemis-depth-FP32-replay-001'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['native','local'],required=True);args=parser.parse_args()
 r=json.loads((DIR/'report.json').read_text());assert r['state']=='passed_every_native_FP32_depth_output_exact_replay_pending_terminal'
 assert r['GPU']=='NVIDIA RTX PRO 6000 Blackwell Server Edition' and r['capability']==[12,0]
 assert r['frozen_inputs_unchanged'] and r['base_runtime_unchanged'] and not r['original_FP64_3e5_criterion_passed']
 lock=json.loads((ROOT/'experiments/artemis-detector-framework/depth-replay-inputs-001.json').read_text())
 assert sha(ROOT/'experiments/artemis-detector-framework/depth-replay-inputs-001.json')==r['input_sha256']
 assert r['base_freeze_after']==lock['base_freeze'] and r['job_id']=='11424868'
 if args.mode=='native':
  assert all(sha(ROOT/p)==v for p,v in lock['frozen_inputs'].items())
  queue=subprocess.check_output(['squeue','-h','-j','11424868','-o','%i %T %N'],text=True)
  accounting=subprocess.check_output(['sacct','-j','11424868','--noheader','--parsable2','--format=JobID,State,ExitCode'],text=True)
  assert not queue.strip() and '11424868|COMPLETED|0:0' in accounting and '11424868.0|COMPLETED|0:0' in accounting
 else:
  terminal=json.loads((DIR/'terminal.json').read_text());assert terminal['actual_terminal'] and terminal['report_sha256']==sha(DIR/'report.json')
  queue=terminal['squeue'];accounting=terminal['sacct'];assert not queue.strip() and '11424868|COMPLETED|0:0' in accounting
 total=0;artifacts={str((DIR/'report.json').relative_to(ROOT)):sha(DIR/'report.json')}
 assert [row['frame'] for row in r['frames']]==['000000','000003']
 for row in r['frames']:
  p=ROOT/row['artifact'];assert sha(p)==row['sha256'];artifacts[row['artifact']]=sha(p)
  with np.load(p,allow_pickle=False) as f:
   assert f.files==['depth_FP32_replay'];a=f['depth_FP32_replay']
  assert np.isfinite(a).all() and str(a.dtype)==row['dtype'] and list(a.shape)==row['shape']
  assert hashlib.sha256(a.tobytes()).hexdigest()==row['replay_array_sha256']
  source=ROOT/'data/engineering/artemis-full-detector-GPU-001'/(row['frame']+'.npz')
  assert sha(source)==row['source_sha256']
  with np.load(source,allow_pickle=False) as f:assert np.array_equal(a,f['depth_pred'])
  assert a.size==row['all_values'] and row['all_values_bit_equal'];total+=a.size
 assert total==798720
 result=dict(state='passed_actual_complete_native_FP32_arithmetic_replay_'+args.mode,actual_terminal=True,
  checked_unix=time.time(),job_id='11424868',report_sha256=sha(DIR/'report.json'),sacct=accounting,squeue=queue,
  all_native_FP32_depth_values_bit_equal=total,artifacts_sha256=artifacts,verifier_sha256=sha(__file__),
  original_FP64_depth_criterion_passed=False,limitation='Same native arithmetic reproduces all saved depth values; original strict FP64 tolerance remains failed, no independent GPU algorithm/task or training claim')
 target=DIR/'terminal.json' if args.mode=='native' else ROOT/'data/provenance/artemis-depth-FP32-replay-local-verification-001.json'
 assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'state':result['state'],'all_values':total}))
if __name__=='__main__':main()
