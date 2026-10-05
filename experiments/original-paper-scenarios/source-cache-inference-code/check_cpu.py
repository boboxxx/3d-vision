"""Real byte/cache conversion, malformed schema and fresh-process input barriers."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

import cache_contract as c


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    assert not args.output.exists();sources=c.sources()
    assert len(c.CONDITIONS)==6 and len(set(c.CONDITIONS))==6
    import torch
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'received.npz'
        left=np.arange(24,dtype=np.uint8).reshape(2,4,3);right=np.flip(left,axis=1).copy();right[1,3,2]=255
        np.savez(path,left=left,right=right)
        row=dict(cache_path=str(path),cache_sha256=c.sha(path),public_hw=[2,4],arrays=dict(left=c.describe(left),right=c.describe(right)))
        values=c.load_pair(row)
        for actual,expected in zip(values,(left,right)):
            assert actual.shape==(1,3,2,4) and actual.dtype==torch.float32 and actual.is_contiguous()
            assert torch.equal(actual,torch.tensor(expected.transpose(2,0,1).copy()).unsqueeze(0).float()/255)
        assert values[1].max()==1
        row['cache_sha256']='0'*64
        try:c.load_pair(row)
        except AssertionError:pass
        else:raise AssertionError('tampered cache accepted')
        for malformed in (np.zeros((2,4,3),np.float32),np.zeros((1,3,2,4),np.uint8),np.zeros((0,4,3),np.uint8)):
            np.savez(path,left=malformed,right=malformed);row.update(cache_sha256=c.sha(path),arrays=dict(left=c.describe(malformed),right=c.describe(malformed)))
            try:c.load_pair(row)
            except AssertionError:pass
            else:raise AssertionError('malformed native pixels accepted')
    code='''import tempfile,sys,numpy as np
from pathlib import Path
import cache_contract as c
with tempfile.TemporaryDirectory() as folder:
 p=Path(folder)/'allowed.npz';np.savez(p,left=np.zeros((2,4,3),np.uint8),right=np.zeros((2,4,3),np.uint8))
 q=Path(folder)/'foreign.npz';q.write_bytes(p.read_bytes())
 control=dict(inference_active=True);sys.addaudithook(c.guard([p],control));assert p.read_bytes()
 for target in (q,Path(folder)/'clean.png',Path('/mnt/d/paper6/data/kitti/training/label_2/000000.txt'),Path(folder)/'weights.pth'):
  try:target.read_bytes()
  except AssertionError:pass
  else:raise AssertionError('forbidden actual read accepted')
 label=Path(folder)/'label_2';label.mkdir();target=label/'000000.txt';target.write_text('post-inference')
 control['inference_active']=False;assert target.read_text()=='post-inference'
print('actual_inference_input_and_post_inference_barriers_passed')'''
    result=subprocess.run([sys.executable,'-c',code],cwd=c.HERE,capture_output=True,text=True)
    assert result.returncode==0 and result.stdout.strip()=='actual_inference_input_and_post_inference_barriers_passed',result.stderr
    for path in c.HERE.glob('*.py'):ast.parse(path.read_text())
    assert c.sources()==sources
    value=dict(state='passed',tests=5,sources=sources,checked_unix=time.time(),protocol_sha256=c.PROTOCOL_SHA,
               families=['six_complete_source_conditions','real_NPZ_to_exact_native_float_tensor',
                         'tampered_hash_and_malformed_cache_rejection','fresh_actual_inference_read_barrier_and_AP_release',
                         'complete_unchanged_native_receiver_and_new_source_identity_syntax'])
    with args.output.open('x') as stream:json.dump(value,stream,indent=2)
    print(json.dumps(dict(state='passed',tests=5)),flush=True)


if __name__=='__main__':main()
