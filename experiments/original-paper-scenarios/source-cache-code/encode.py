"""Seal every real source wire; independent receivers later consume wire only."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

import common as c
sys.path.insert(0, str(c.HERE.parent / 'code'))
from image_codecs import encode_pair


def initialize(allowed):
    c.native_environment(); sys.addaudithook(c.guard('source', allowed))


def encode_one(job):
    key, codec, rate, parameter, frame, directory = job
    paths = [c.DATA / 'training' / view / (frame + '.png') for view in ('image_2', 'image_3')]
    originals = {str(p): c.sha(p) for p in paths}; pair = []
    for p in paths:
        with Image.open(p) as image:
            assert image.mode == 'RGB'; pair.append(c.hwc(np.array(image, dtype=np.uint8)))
    wire, _, evidence = encode_pair(*pair, codec=codec, parameter=parameter)
    target = Path(directory) / 'source' / key / (frame + '.p6sb')
    with target.open('xb') as stream: stream.write(wire)
    assert c.sha(target) == evidence['wire_sha256'] and all(c.sha(p) == h for p, h in originals.items())
    return dict(frame_id=frame, condition=key, codec=codec, nominal_compression=rate, parameter=parameter,
                source_images_sha256=originals, wire_path=str(target), evidence=evidence)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--scope', choices=('engineering', 'main'), required=True); args = parser.parse_args()
    directory = args.directory.resolve(); c.native_environment()
    assert not directory.exists()
    sources = c.identities(); previous = c.predecessor(); ids = c.frames(args.scope)
    allowed = [c.DATA / 'training' / view / (frame + '.png') for frame in ids for view in ('image_2', 'image_3')]
    sys.addaudithook(c.guard('source', allowed))
    directory.mkdir(parents=True)
    for key, *_ in c.CONDITIONS: (directory / 'source' / key).mkdir(parents=True)
    output = directory / 'encode.json'; progress = directory / 'encoding-progress.json'
    info = dict(state='running', pid=os.getpid(), started_unix=time.time(), scope=args.scope, directory=str(directory),
        frame_ids=ids, protocol_sha256=c.PROTOCOL_SHA, sources=sources, predecessor=previous, environment=c.environment(),
        workers=4, conditions={}, actual_encodes=0, source_only_PHY_uses=None, source_only_PHY_energy=None,
        input_barrier=True, no_GT_model_AP_or_radio=True)
    c.save(output, info)
    try:
        with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn'), initializer=initialize, initargs=(allowed,)) as pool:
            for key, codec, rate, parameter in c.CONDITIONS:
                records = []
                jobs = [(key, codec, rate, parameter, frame, str(directory)) for frame in ids]
                for record in pool.map(encode_one, jobs, chunksize=1):
                    records.append(record); info['actual_encodes'] += 1
                    if len(records) % 100 == 0:
                        c.save(progress, dict(state='running', condition=key, completed_in_condition=len(records), actual_encodes=info['actual_encodes']))
                        print(json.dumps(dict(condition=key, encoded=len(records))), flush=True)
                assert len(records) == len(ids)
                info['conditions'][key] = dict(codec=codec, nominal_compression=rate, parameter=parameter, records=records)
                c.save(output, info)
        assert info['actual_encodes'] == 6 * len(ids) and c.identities() == sources and c.predecessor() == previous
        c.native_environment()
        info.update(state='finished_all_six_complete_source_wires', ended_unix=time.time())
        c.save(progress, dict(state='finished', actual_encodes=info['actual_encodes']))
    except BaseException as error:
        info.update(state='failed', exception=repr(error), ended_unix=time.time()); raise
    finally: c.save(output, info)
    print(json.dumps(dict(state=info['state'], pairs=info['actual_encodes'])), flush=True)


if __name__ == '__main__': main()
