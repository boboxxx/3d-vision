"""Actual wire/header-only native float32 reconstruction with frozen SRCNN002."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch

import common as c


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--scope', choices=('engineering', 'main'), required=True)
    parser.add_argument('--rate', type=int, choices=(10, 30, 50), required=True); args = parser.parse_args()
    source = c.CPU_gate(); runs, dependencies = c.gate(args.scope); free = c.cuda_gate()
    prefix = c.prefix(args.scope); base = c.NATIVE / prefix; parent_path = c.ROOT / 'data/runs' / (prefix + '-encode.json')
    parent = c.read(parent_path)
    assert parent['state'] == 'finished_all_three_source_wire_conditions' and parent['sources'] == source and parent['dependencies'] == dependencies
    assert parent['frame_ids'] == c.ids(args.scope)
    # Routing and expected byte identity only, no sender native geometry/arrays.
    routes = [(row['frame_id'], row['wire_path'], row['wire_sha256']) for row in parent['conditions'][str(args.rate)]['rows']]
    assert [v[0] for v in routes] == parent['frame_ids']
    folder = base / ('cr' + str(args.rate)) / 'received'; manifest = c.ROOT / 'data/runs' / (prefix + '-cr' + str(args.rate) + '-receive.json')
    assert not folder.exists() and not manifest.exists()
    selected = runs[args.rate]; training_scope = 'engineering' if args.scope == 'engineering' else 'formal'
    model = c.r.model_from_checkpoint(Path(selected['checkpoint']).read_bytes(), selected, source['training'], training_scope, args.rate, 'cuda')
    initial = c.r.states(model); folder.mkdir(); torch.cuda.reset_peak_memory_stats()
    control = {'active': False}; sys.addaudithook(c.r.guard([v[1] for v in routes], control))
    run = dict(state='running', pid=os.getpid(), scope=args.scope, rate=args.rate, sources=source, dependencies=dependencies,
               encoding_manifest_sha256=c.sha(parent_path), ordered_ids=[v[0] for v in routes], frames=[],
               checkpoint_sha256=selected['checkpoint_sha256'], model_initial_state=initial,
               physical_free_before_bytes=free, started_unix=time.time(), data_barrier=True,
               TF32=False, mixed_precision=False, CPU_threads=2, cudnn_deterministic=True,
               received_header_only_geometry=True, no_clean_GT_calibration_or_detector=True,
               source_only_PHY_uses=None, source_only_PHY_energy=None)
    c.save(manifest, run)
    try:
        control['active'] = True
        with torch.no_grad():
            for frame, path, digest in routes:
                wire = Path(path).read_bytes(); assert c.r.sha_bytes(wire) == digest
                output, row = c.r.receive(wire, model); assert row['nominal_compression'] == args.rate
                blob = c.r.pack_pair(output); cache = folder / (frame + '.npz')
                with cache.open('xb') as stream: stream.write(blob)
                run['frames'].append(dict(frame_id=frame, wire_path=path, cache_path=str(cache), cache_sha256=c.r.sha_bytes(blob), **row))
                assert torch.cuda.max_memory_reserved() + 2 * 2**30 <= free
                if len(run['frames']) % 100 == 0: c.save(manifest, run)
        control['active'] = False; assert c.r.states(model) == initial and c.sources() == source
        run.update(state='finished_all_native_received_pairs', ended_unix=time.time(), model_final_state=c.r.states(model),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:
        control['active'] = False; run.update(state='failed', error=repr(error), ended_unix=time.time()); raise
    finally: c.save(manifest, run)
    print(json.dumps(dict(state=run['state'], rate=args.rate, pairs=len(run['frames']))), flush=True)


if __name__ == '__main__': main()
