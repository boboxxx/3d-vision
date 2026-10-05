"""Unique sealed eight-check CPU gate for the current geometry/native transport sources."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import unittest
import numpy as np
import torch
from check_cpu import GeometryChecks
from check_native_transport_cpu import TransportChecks


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    assert not torch.cuda.is_available(), 'this gate must not initialize an active GPU'
    assert not args.output.exists(), 'preserve earlier evidence'
    root=Path(__file__).resolve().parents[3]
    paths=list(Path(__file__).parent.glob('*.py'))+[root/'experiments/geometry-risk/native-engineering-001.md',root/'experiments/geometry-risk/native-engineering-inputs-001.json',root/'experiments/geometry-risk/protocol-001.md']
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(GeometryChecks),unittest.defaultTestLoader.loadTestsFromTestCase(TransportChecks)])
    started=time.time();result=unittest.TextTestRunner(verbosity=2).run(suite)
    assert hashes=={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    record=dict(state='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,numpy=np.__version__,torch=torch.__version__,CUDA_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                started_unix=started,ended_unix=time.time(),source_sha256=hashes,scope='eight_synthetic_CPU_checks_no_native_task_calibration_or_AP')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(record,indent=2)+'\n')
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
