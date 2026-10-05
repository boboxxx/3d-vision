"""Once-only three-rate sensor-side actual PNG to SRCNN wire files, CPU only."""
import argparse
from pathlib import Path
import sys
import time

import common as c


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--scope', choices=('engineering', 'main'), required=True)
    args = parser.parse_args(); source = c.CPU_gate(); _, dependencies = c.gate(args.scope)
    frames = c.ids(args.scope); prefix = c.prefix(args.scope)
    directory = c.NATIVE / prefix; manifest = c.ROOT / 'data/runs' / (prefix + '-encode.json')
    assert not directory.exists() and not manifest.exists()
    import os
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only encoder required'
    paths = [c.DATA / 'training' / camera / (frame + '.png') for frame in frames for camera in ('image_2', 'image_3')]
    control = {'active': False}; sys.addaudithook(c.sensor_guard(paths, control)); directory.mkdir()
    run = dict(state='running', pid=os.getpid(), scope=args.scope, frame_ids=frames, sources=source, dependencies=dependencies,
               started_unix=time.time(), conditions={}, source_only_PHY_uses=None, source_only_PHY_energy=None,
               no_GT_calibration_LiDAR_or_detector=True, GPU_used=False)
    c.save(manifest, run)
    try:
        control['active'] = True
        for rate in c.r.interface.RATES:
            folder = directory / ('cr' + str(rate)) / 'wire'; folder.mkdir(parents=True)
            rows = []; raw_total = wire_total = 0
            for frame in frames:
                left, l = c.png(c.DATA / 'training/image_2' / (frame + '.png'))
                right, r = c.png(c.DATA / 'training/image_3' / (frame + '.png'))
                wire, meta = c.r.interface.encode(left, right, rate); path = folder / (frame + '.p6sr')
                with path.open('xb') as stream: stream.write(wire)
                rows.append(dict(frame_id=frame, wire_path=str(path), wire_sha256=c.r.sha_bytes(wire),
                                 source_left=l, source_right=r, source_metadata=meta))
                raw_total += left.size + right.size; wire_total += len(wire)
            run['conditions'][str(rate)] = dict(rows=rows, raw_RGB8_bytes=raw_total, wire_bytes=wire_total,
                                               actual_pooled_raw_to_wire_ratio=raw_total / wire_total)
            c.save(manifest, run)
        control['active'] = False; assert c.sources() == source
        run.update(state='finished_all_three_source_wire_conditions', ended_unix=time.time(), pairs=3 * len(frames))
    except BaseException as error:
        control['active'] = False; run.update(state='failed', error=repr(error), ended_unix=time.time()); raise
    finally: c.save(manifest, run)
    print(__import__('json').dumps(dict(state=run['state'], pairs=run['pairs'])), flush=True)


if __name__ == '__main__': main()
