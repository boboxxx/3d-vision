"""Five execution-contract families using actual discarded SRCNN002 weights."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch

import common as c
import audit as a


def reject(action):
    try: action()
    except (AssertionError, ValueError): return
    raise AssertionError('Invalid input was accepted')


def local_checkpoint(path):
    value = Path(path)
    if c.ROOT == Path('/home/sheng/paper6'): return value
    assert value.is_relative_to('/mnt/d/paper6/runs')
    return c.ROOT / 'data/engineering/srcnn-kitti-engineering-seed17-002-native-transfer' / str(value).lstrip('/')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--host', choices=('local', 'sheng'), required=True)
    args = parser.parse_args(); torch.set_num_threads(2); source = c.sources()
    output = c.ROOT / 'data/engineering' / ('srcnn-native-cache-' + args.host + '-CPU-002.json'); assert not output.exists()
    assert c.sha(c.r.PROTOCOL) == c.r.PROTOCOL_SHA
    frames = [np.random.default_rng(1703).integers(0, 256, (33, 49, 3), dtype=np.uint8)]
    left = frames[0]; right = np.flip(left, axis=1).copy()
    records = []; maximum_error = 0.; overshoot_views = 0; corrupt_checks = 0
    for rate in (10, 30, 50):
        path = c.ROOT / 'data/runs' / ('srcnn-kitti-engineering-seed17-002-cr' + str(rate) + '.json')
        run = c.read(path); blob = local_checkpoint(run['checkpoint']).read_bytes()
        model = c.r.model_from_checkpoint(blob, run, source['training'], 'engineering', rate, 'cpu')
        initial = c.r.states(model); wire, meta = c.r.interface.encode(left, right, rate)
        assert wire == a.independent_wire(left, right, rate)
        views, shape, received_rate = a.independent_unpack(wire)
        assert shape == (33, 49) and received_rate == rate and len(views) == 2
        assert meta['source_bytes'] == len(wire) and meta['framing_bytes'] == 32
        with torch.no_grad():
            actual, row = c.r.receive(wire, model)
            expected = a.independent_receive(wire, model.state_dict())
        packed = c.r.pack_pair(actual); row['cache_sha256'] = c.r.sha_bytes(packed)
        tensors = c.r.tensor_pair(packed, row); errors = []
        for index, value in enumerate(actual):
            error = float(np.max(np.abs(value.astype(np.float64) - expected[index].astype(np.float64))))
            assert error <= 3e-5 and value.shape == left.shape and value.dtype == np.dtype('<f4')
            assert np.array_equal(tensors[index][0].numpy().transpose(1, 2, 0), value)
            errors.append(error); maximum_error = max(maximum_error, error)
            overshoot_views += bool(np.any((value < 0) | (value > 1)))
        assert c.r.states(model) == initial
        bad_run = dict(run, checkpoint_sha256='0' * 64)
        reject(lambda: c.r.model_from_checkpoint(blob, bad_run, source['training'], 'engineering', rate))
        reject(lambda: c.r.model_from_checkpoint(blob, run, source['training'], 'formal', rate))
        reject(lambda: c.r.model_from_checkpoint(blob, run, {}, 'engineering', rate))
        reject(lambda: c.r.model_from_checkpoint(blob, run, source['training'], 'engineering', 30 if rate == 10 else 10))
        corrupt = [b'BAD!' + wire[4:], wire[:7] + b'\x01' + wire[8:], wire[:-1], wire[:-1] + bytes([wire[-1] ^ 1])]
        for bad in corrupt:
            reject(lambda: a.independent_unpack(bad)); reject(lambda: c.r.interface.unpack(bad)); corrupt_checks += 2
        reject(lambda: c.r.tensor_pair(packed[:-1], row))
        records.append(dict(rate=rate, checkpoint_sha256=run['checkpoint_sha256'], wire_sha256=c.r.sha_bytes(wire),
                            source_metadata=meta, readonly_states=initial, arrays=row['arrays'],
                            maximum_errors=errors, ranges=row['ranges']))
    # Storage must preserve overshoot even when these trained random fixtures
    # happen to remain in range. This does not alter or select model outputs.
    extremes = [np.array([[[-.25, 1.25, .3], [2., -2., .5]]], dtype='<f4') for _ in range(2)]
    packed_extremes = c.r.pack_pair(extremes)
    extreme_row = dict(cache_sha256=c.r.sha_bytes(packed_extremes), native_hw=[1, 2],
                       arrays={key:c.r.describe(value) for key,value in zip(('left','right'), extremes)})
    extreme_tensors = c.r.tensor_pair(packed_extremes, extreme_row)
    assert all(np.array_equal(t[0].numpy().transpose(1, 2, 0), value) for t,value in zip(extreme_tensors, extremes))
    assert all(float(t.min()) == -2 and float(t.max()) == 2 for t in extreme_tensors)
    # Real open events; static model loading already finished before these hooks.
    with tempfile.TemporaryDirectory() as folder:
        base = Path(folder); sensor = base / 'source.png'; wire_file = base / 'selected.p6sr'; foreign = base / 'foreign.p6sr'
        for path in (sensor, wire_file, foreign): path.write_bytes(b'fixture')
        sensor_control = {'active': False}; receive_control = {'active': False}
        sys.addaudithook(c.sensor_guard([sensor, wire_file], sensor_control))
        sys.addaudithook(c.r.guard([wire_file], receive_control))
        sensor_control['active'] = True; assert sensor.read_bytes() == wire_file.read_bytes() == b'fixture'
        reject(foreign.read_bytes); sensor_control['active'] = False
        receive_control['active'] = True; assert wire_file.read_bytes() == b'fixture'
        reject(sensor.read_bytes); reject(foreign.read_bytes); receive_control['active'] = False
    assert c.sources() == source
    result = dict(state='passed_five_native_SRCNN_cache_CPU_families', checked_unix=time.time(), host=args.host,
                  families=5, source_geometry=[33, 49], views=6, RGB_values=6 * 33 * 49 * 3,
                  sources=source, records=records, maximum_RGB_error=maximum_error, model_overshoot_views=overshoot_views,
                  guaranteed_overshoot_cache_views=2, guaranteed_overshoot_RGB_values=12,
                  malformed_header_CRC_checks=corrupt_checks, actual_file_open_barriers_checked=True,
                  limitation='Synthetic execution contract with actual discarded six-update weights; no native KITTI/GPU reconstruction, formal model eligibility, quality or AP')
    with output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k:result[k] for k in ('state', 'families', 'views', 'maximum_RGB_error', 'model_overshoot_views', 'guaranteed_overshoot_cache_views')}))


if __name__ == '__main__': main()
