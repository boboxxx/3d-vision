"""Explicitly simulated tar transport of real wires and synthetic full-size NPZ."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VERIFIER = ROOT / 'data/provenance/verify-srcnn-native-cache-local-002.py'
spec = importlib.util.spec_from_file_location('streamed_acceptance', VERIFIER)
v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)

def archive(items):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as out:
        for name, blob, kind in items:
            info = tarfile.TarInfo(name.lstrip('/'))
            if kind == 'symlink': info.type = tarfile.SYMTYPE; info.linkname = 'foreign'
            elif kind == 'oversized': info.size = v.transport.MAX_MEMBER_BYTES + 1
            else: info.size = len(blob)
            out.addfile(info, io.BytesIO(blob) if kind == 'regular' else None)
    return buffer.getvalue()

class MemoryProcess:
    def __init__(self, blob, code):
        self.stdin = io.BytesIO(); self.stdout = io.BytesIO(blob); self.stderr = io.BytesIO()
        self.code = code
    def wait(self, timeout=None): return self.code
    def poll(self): return self.code
    def kill(self): raise AssertionError('unexpected kill in memory fixture')

def main():
    output = ROOT / 'data/engineering/srcnn-streamed-cache-CPU-002.json'; assert not output.exists()
    encode = v.c.read(ROOT / 'data/runs/srcnn-source-cache-engineering-001-encode.json')
    proof_path = ROOT / 'data/provenance/srcnn-source-cache-engineering-001-wire-local-verification.json'
    proof = v.c.read(proof_path)
    assert proof['state'] == 'passed_all6_actual_transferred_SRCNN_engineering_wire_files_and8_artifacts'
    tests = []; pixels = 0; blob_bytes = 0
    for rate in (10, 30, 50):
        prefix = 'srcnn-source-cache-main-001'; ids = []; receipt = {'frames': []}
        closure = {'artifacts_sha256': {}}; mapping = {}; items = []; cache_values = {}
        for source in encode['conditions'][str(rate)]['rows']:
            frame = source['frame_id']; ids.append(frame)
            wire_path = ROOT / 'data/engineering/srcnn-source-cache-engineering-001-wire-transfer' / source['wire_path'].lstrip('/')
            wire = wire_path.read_bytes(); assert v.c.r.sha_bytes(wire) == source['wire_sha256']
            _, h, w, *_ = v.literal_header(wire)
            values = [np.broadcast_to(np.array([-2., .5, 2.], dtype='<f4'), (h,w,3)).copy(),
                      np.broadcast_to(np.array([2., -1., .25], dtype='<f4'), (h,w,3)).copy()]
            blob = v.c.r.pack_pair(values)
            base = str(v.c.NATIVE / prefix / ('cr' + str(rate)))
            wire_name = base + '/wire/' + frame + '.p6sr'; cache_name = base + '/received/' + frame + '.npz'
            row = dict(frame_id=frame, wire_path=wire_name, cache_path=cache_name,
                wire_sha256=v.c.r.sha_bytes(wire), cache_sha256=v.c.r.sha_bytes(blob), native_hw=[h,w],
                arrays={name:v.c.r.describe(a) for name,a in zip(('left','right'),values)},
                ranges=[dict(minimum=float(a.min()),maximum=float(a.max()),out_of_range_fraction=float(np.mean((a<0)|(a>1)))) for a in values])
            receipt['frames'].append(row); cache_values[cache_name] = values
            for name, data in ((wire_name,wire),(cache_name,blob)):
                relative = 'data/engineering/' + prefix + '-native-transfer/' + name.lstrip('/')
                mapping[relative] = name; closure['artifacts_sha256'][relative] = v.c.r.sha_bytes(data)
                items.append((name,data,'regular'))
        def attempt(label, members, modified_receipt=None, modified_closure=None, code=0, reject=False):
            packed = archive(members)
            def fake(cmd, **kwargs):
                assert cmd == v.transport.SSH + ['cd / && tar --create --file=- --no-recursion --verbatim-files-from --files-from=-']
                return MemoryProcess(packed,code)
            with patch.object(v.transport.subprocess,'Popen',side_effect=fake):
                try:
                    result = v.stream_rate(prefix,rate,ids,modified_receipt or receipt,modified_closure or closure,mapping)
                except AssertionError:
                    assert reject, label
                    tests.append(dict(rate=rate,case=label,result='rejected'))
                    return
                assert not reject, label + ': unexpectedly accepted'
                tests.append(dict(rate=rate,case=label,result='passed'))
                return result
        headers, counts, lengths, rows = attempt('complete_real_wire_synthetic_unclipped_NPZ',items)
        assert len(rows)==4 and sum(counts.values())==sum(a.size for values in cache_values.values() for a in values)
        pixels += sum(counts.values()); blob_bytes += sum(lengths.values())
        attempt('duplicate',items+[items[0]],reject=True)
        attempt('missing',items[:-1],reject=True)
        attempt('foreign',items+[('/foreign/file',b'x','regular')],reject=True)
        attempt('nonregular',[(items[0][0],b'','symlink')]+items[1:],reject=True)
        attempt('oversized',[(items[0][0],b'','oversized')],reject=True)
        attempt('nonzero_remote_exit',items,code=2,reject=True)
        bad = [(items[0][0], items[0][1][:-1]+bytes([items[0][1][-1]^1]),'regular')]+items[1:]
        attempt('SHA_mismatch',bad,reject=True)
        # Update declared hashes so CRC/pixel/dtype checks, not only SHA, decide.
        altered_receipt=copy.deepcopy(receipt); altered_closure=copy.deepcopy(closure)
        altered_receipt['frames'][0]['wire_sha256']=v.c.r.sha_bytes(bad[0][1])
        rel='data/engineering/'+prefix+'-native-transfer/'+bad[0][0].lstrip('/')
        altered_closure['artifacts_sha256'][rel]=v.c.r.sha_bytes(bad[0][1])
        attempt('CRC_under_updated_SHA',bad,altered_receipt,altered_closure,reject=True)
        for label, convert in [('clipped_values',lambda a:a.clip(0,1)),('float64_dtype',lambda a:a.astype('<f8'))]:
            altered_receipt=copy.deepcopy(receipt); altered_closure=copy.deepcopy(closure)
            target=items[1][0]; buf=io.BytesIO(); values=cache_values[target]
            np.savez(buf,left=convert(values[0]),right=convert(values[1])); blob=buf.getvalue()
            members=list(items);members[1]=(target,blob,'regular')
            altered_receipt['frames'][0]['cache_sha256']=v.c.r.sha_bytes(blob)
            rel='data/engineering/'+prefix+'-native-transfer/'+target.lstrip('/')
            altered_closure['artifacts_sha256'][rel]=v.c.r.sha_bytes(blob)
            attempt(label+'_under_updated_SHA',members,altered_receipt,altered_closure,reject=True)
        altered_receipt=copy.deepcopy(receipt); altered_receipt['frames'][0]['ranges'][0]['minimum']=0.
        attempt('range_metadata',items,altered_receipt,reject=True)
    result=dict(state='passed_simulated_tar_real6_wires_synthetic6_full_geometry_float_caches',
        checked_unix=time.time(),actual_engineering_wires=6,synthetic_float_pairs=6,
        synthetic_RGB_values=pixels,fixture_blob_bytes=blob_bytes,tests=tests,
        rejected_cases=sum(t['result']=='rejected' for t in tests),
        verifier_sha256=v.c.sha(VERIFIER),transport_sha256=v.c.sha(v.HELPER),check_sha256=v.c.sha(__file__),
        wire_local_proof_sha256=v.c.sha(proof_path),
        protocol_sha256=v.c.sha(ROOT/'experiments/original-paper-scenarios/srcnn-streamed-cache-CPU-acceptance-002.md'),
        retained_blob_files=0,
        limitation='Explicit in-memory subprocess simulation: real six engineering wires, synthetic unclipped NPZ; no actual remote NPZ, main closure, GPU reconstruction, formal weights, quality or detection')
    with output.open('x') as out: json.dump(result,out,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ['state','synthetic_RGB_values','rejected_cases']}))

if __name__=='__main__':main()
