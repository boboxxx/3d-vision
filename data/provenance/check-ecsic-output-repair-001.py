"""Actual output helper works while every NPZ read remains forbidden in child."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
CODE=ROOT/'experiments/original-paper-scenarios/ecsic-entropy-code'


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.worker:
        sys.path.insert(0,str(CODE))
        from native import save_arrays_without_readback
        blocked=[]
        def guard(event,values):
            if event=='open' and isinstance(values[0],(str,bytes)):
                name=os.fsdecode(values[0]);mode=values[1];flags=values[2]
                read=(isinstance(mode,str) and 'r' in mode) or (mode is None and flags & os.O_ACCMODE==os.O_RDONLY)
                if read and name.endswith('.npz'):
                    blocked.append(name);raise RuntimeError('NPZ read forbidden')
        sys.addaudithook(guard)
        arrays=dict(integers=np.arange(100,dtype=np.int32),float_values=np.linspace(-1,1,21,dtype=np.float32))
        digest=save_arrays_without_readback(args.worker,arrays)
        try:args.worker.read_bytes()
        except RuntimeError:pass
        else:raise AssertionError('input barrier not active')
        assert len(blocked)==1
        print(json.dumps(dict(digest=digest,blocked_reads=len(blocked))));return
    assert args.output and not args.output.exists()
    with tempfile.TemporaryDirectory(prefix='ecsic-output-') as d:
        path=Path(d)/'output.npz'
        r=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(path)],capture_output=True,text=True)
        assert r.returncode==0,r.stderr
        result=json.loads(r.stdout);assert result['digest']==hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path,allow_pickle=False) as data:
            assert np.array_equal(data['integers'],np.arange(100,dtype=np.int32))
            assert np.array_equal(data['float_values'],np.linspace(-1,1,21,dtype=np.float32))
    result.update(state='passed',native_source_sha256=hashlib.sha256((CODE/'native.py').read_bytes()).hexdigest(),
                  verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
