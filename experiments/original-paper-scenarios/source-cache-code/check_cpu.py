"""Four structural/real input-barrier checks before native engineering."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import common as c


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    assert not args.output.exists()
    sources = c.identities(); previous = c.predecessor()
    assert c.CONDITIONS == [('jpeg-cr10', 'jpeg', 10, 90), ('jpeg-cr30', 'jpeg', 30, 39), ('jpeg-cr50', 'jpeg', 50, 17),
                            ('jpeg2000-cr10', 'jpeg2000', 10, 10), ('jpeg2000-cr30', 'jpeg2000', 30, 30), ('jpeg2000-cr50', 'jpeg2000', 50, 50)]
    assert c.frames('engineering') == ['000000', '000003']
    c.hwc(np.zeros((2, 4, 3), np.uint8))
    for bad in (np.zeros((2, 4, 3), np.float32), np.zeros((1, 3, 2, 4), np.uint8), np.zeros((0, 4, 3), np.uint8)):
        try: c.hwc(bad)
        except AssertionError: pass
        else: raise AssertionError('malformed received schema accepted')
    # Actual fresh interpreter opens, not manual callback invocation.
    code = '''import io,sys,tempfile,numpy as np
from pathlib import Path
from PIL import Image
import common as c
with tempfile.TemporaryDirectory() as d:
 p=Path(d)/'clean.png'; Image.fromarray(np.zeros((2,4,3),np.uint8)).save(p)
 q=Path(d)/'foreign.npz'; np.savez(q,a=np.zeros(1))
 sys.addaudithook(c.guard('receiver'))
 for path in (p,q,Path('/mnt/d/paper6/data/kitti/training/label_2/000000.txt')):
  try: path.read_bytes()
  except AssertionError: pass
  else: raise AssertionError('actual forbidden read succeeded')
 buffer=io.BytesIO();np.savez(buffer,left=np.zeros((2,4,3),np.uint8),right=np.zeros((2,4,3),np.uint8));assert buffer.getvalue()
print('actual_receiver_barriers_and_memory_writer_passed')'''
    result = subprocess.run([sys.executable, '-c', code], cwd=c.HERE, capture_output=True, text=True)
    assert result.returncode == 0 and result.stdout.strip() == 'actual_receiver_barriers_and_memory_writer_passed', result.stderr
    for p in c.HERE.glob('*.py'): ast.parse(p.read_text())
    assert c.identities() == sources and c.predecessor() == previous
    value = dict(state='passed', tests=4, checked_unix=time.time(), sources=sources, predecessor=previous,
        protocol_sha256=c.PROTOCOL_SHA, families=['fixed_six_conditions_and_training_ids', 'received_HWC_uint8_schema',
        'actual_fresh_process_PNG_GT_NPZ_read_barriers_and_memory_output', 'locked_closed_predecessor_source_identity_and_syntax'])
    with args.output.open('x') as stream: json.dump(value, stream, indent=2)
    print(json.dumps(dict(state='passed', tests=4)), flush=True)


if __name__ == '__main__': main()
