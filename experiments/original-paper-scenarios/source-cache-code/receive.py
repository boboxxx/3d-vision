"""Fresh wire-only whole-condition decoder and fixed native uint8 image cache."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

import common as c
sys.path.insert(0, str(c.HERE.parent / 'code'))
from image_codecs import decode_pair
from contracts import digest_array


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--condition', choices=tuple(x[0] for x in c.CONDITIONS), required=True)
    parser.add_argument('--scope', choices=('engineering', 'main'), required=True); args = parser.parse_args()
    c.native_environment(); ids = c.frames(args.scope); sources = c.identities()
    directory = args.directory.resolve(); source = directory / 'source' / args.condition
    assert {p.name for p in source.iterdir()} == {x + '.p6sb' for x in ids}
    target = directory / 'received' / args.condition; assert not target.exists()
    sys.addaudithook(c.guard('receiver'))
    target.mkdir(parents=True); output = target / 'receiver.json'
    info = dict(state='running', pid=os.getpid(), started_unix=time.time(), scope=args.scope, condition=args.condition,
        environment=c.environment(), sources=sources, protocol_sha256=c.PROTOCOL_SHA, frame_ids=ids, records=[],
        input_barrier=True, PNG_or_source_manifest_inputs=False, dimensions_from_received_stream_headers=True,
        HWC_uint8_no_resize=True, no_model_AP_or_radio=True)
    c.save(output, info)
    try:
        for index, frame in enumerate(ids, 1):
            wire_path = source / (frame + '.p6sb'); wire = wire_path.read_bytes()
            left, right = (c.hwc(x) for x in decode_pair(wire))
            assert left.shape == right.shape
            buffer = io.BytesIO(); np.savez(buffer, left=left, right=right); payload = buffer.getvalue()
            path = target / (frame + '.npz')
            with path.open('xb') as stream: stream.write(payload)
            info['records'].append(dict(frame_id=frame, wire_path=str(wire_path), wire_sha256=hashlib.sha256(wire).hexdigest(),
                cache_path=str(path), cache_sha256=hashlib.sha256(payload).hexdigest(),
                public_hw=list(left.shape[:2]), arrays=dict(left=digest_array(left), right=digest_array(right))))
            if index % 100 == 0:
                print(json.dumps(dict(condition=args.condition, decoded=index)), flush=True)
        assert len(info['records']) == len(ids) and c.identities() == sources
        c.native_environment(); info.update(state='finished_all_received_pairs', ended_unix=time.time())
    except BaseException as error:
        info.update(state='failed', exception=repr(error), ended_unix=time.time()); raise
    finally: c.save(output, info)
    print(json.dumps(dict(state=info['state'], condition=args.condition, pairs=len(ids))), flush=True)


if __name__ == '__main__': main()
