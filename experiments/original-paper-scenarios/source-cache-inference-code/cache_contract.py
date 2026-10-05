"""Sealed shared native cache contract; original receiver code remains unchanged."""
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NATIVE = Path('/mnt/d/paper6/runs')
DATA = Path('/mnt/d/paper6/data/kitti')
LEGACY = ROOT / 'experiments/original-detector-integration/final-code'
PROTOCOL = HERE.parent / 'source-cache-inference-protocol-001.md'
PROTOCOL_SHA = '192e54424382183b38ef87731c2e5bd96eed6ceae31eeccffd897e6c040a6b08'
CONDITIONS = tuple(f'{codec}-cr{rate}' for codec in ('jpeg','jpeg2000') for rate in (10,30,50))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()


def read(p): return json.loads(Path(p).read_text())
def save(p,v):
    p=Path(p); temporary=p.with_suffix('.tmp')
    temporary.write_text(json.dumps(v,indent=2,allow_nan=False)); temporary.replace(p)


def sources():
    assert sha(PROTOCOL)==PROTOCOL_SHA
    sys.path.insert(0,str(LEGACY))
    import common as legacy
    current={str(p.relative_to(ROOT)):sha(p) for p in sorted(HERE.glob('*.py'))}
    assert len(current)==4
    return dict(native=legacy.current_sources(),inference=current,protocol_sha256=PROTOCOL_SHA)


def proof(scope):
    prefix='original-source-cache-'+scope+'-001'
    closure=read(ROOT/'data/provenance'/(prefix+'-closure.json'))
    local=read(ROOT/'data/provenance'/(prefix+'-local-verification.json'))
    assert closure['state']=='closed_actual_terminal_all_native_artifacts' and closure['actual_terminal']
    assert local['closure_sha256']==sha(ROOT/'data/provenance'/(prefix+'-closure.json'))
    assert closure['pairs']==local['pairs']==(12 if scope=='engineering' else 22614)
    assert local['state']==('passed_all_transferred_records_and_engineering_wires' if scope=='engineering' else
                            'passed_all22614_transferred_pair_records_and_native_audit_metadata')
    for group in closure['sources'].values():
        for p,h in group.items(): assert sha(ROOT/p)==h
    directory=NATIVE/prefix
    assert sha(directory/'encode.json')==closure['native_files'][str(directory/'encode.json')]
    encoding=read(directory/'encode.json')
    ids=encoding['frame_ids']; assert len(ids)==(2 if scope=='engineering' else 3769)
    assert len(ids)==len(set(ids))
    if scope=='main':
        split=DATA/'ImageSets/val.txt'
        assert sha(split)=='657ac4bcc1e156e5b106a4ca18e1f88e012787ea1d2b5d0adeea97fee903fa86'
        assert split.read_text().splitlines()==ids
    else: assert ids==['000000','000003']
    f9=read(ROOT/'data/provenance/stereo-epipolar-native-seed17-001-closure.json')
    f9local=read(ROOT/'data/provenance/stereo-epipolar-native-seed17-001-local-verification-001.json')
    assert f9['actual_terminal'] and f9local['closure_sha256']==sha(ROOT/'data/provenance/stereo-epipolar-native-seed17-001-closure.json')
    assert f9local['state']=='passed_all_transferred_artifacts13360_training_rows16_final_endpoints'
    return directory,ids,closure,local


def describe(a):
    a=np.ascontiguousarray(a); value=dict(dtype=a.dtype.str,shape=list(a.shape))
    return dict(**value,sha256=hashlib.sha256(json.dumps(value,sort_keys=True).encode()+b'\0'+a.tobytes()).hexdigest())


def load_pair(row):
    path=Path(row['cache_path']); blob=path.read_bytes()
    assert hashlib.sha256(blob).hexdigest()==row['cache_sha256']
    arrays=[]
    with np.load(io.BytesIO(blob),allow_pickle=False) as values:
        assert set(values.files)=={'left','right'}
        for name in ('left','right'):
            a=values[name]
            assert a.dtype==np.uint8 and a.ndim==3 and a.shape[2]==3 and all(x>0 for x in a.shape)
            assert describe(a)==row['arrays'][name]
            arrays.append(np.ascontiguousarray(a))
    assert arrays[0].shape==arrays[1].shape and list(arrays[0].shape[:2])==row['public_hw']
    import torch
    return tuple(torch.from_numpy(a.transpose(2,0,1).copy()).unsqueeze(0).float().div(255) for a in arrays)


def guard(allowed_npz, control):
    allowed={str(Path(p).resolve()) for p in allowed_npz}
    def barrier(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)): return
        name=os.fsdecode(args[0]);mode,flags=args[1:3]
        reading=(isinstance(mode,str) and 'r' in mode) or (mode is None and flags & os.O_ACCMODE==os.O_RDONLY)
        if not reading or not control['inference_active']: return
        assert not name.lower().endswith(('.png','.pkl','.pth','.pt','.bin')), 'clean/weights/infos/LiDAR inference read forbidden'
        assert '/label_2/' not in name and '/velodyne/' not in name, 'GT/LiDAR inference read forbidden'
        if name.endswith('.npz'): assert str(Path(name).resolve()) in allowed, 'foreign cache input forbidden'
    return barrier
