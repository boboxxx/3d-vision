"""Full independent framing/source/color and direct six-array convolution audit."""
import argparse
import io
import json
import math
from pathlib import Path
import struct
import sys
import time
import zlib

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

import common as c


def independent_wire(left, right, rate):
    h, w = left.shape[:2]; lh, lw = math.floor(h / math.sqrt(rate)), math.floor(w / math.sqrt(rate))
    payloads = [np.asarray(Image.fromarray(a).resize((lw, lh), Image.Resampling.BICUBIC), dtype=np.uint8).tobytes() for a in (left, right)]
    payload = b''.join(payloads)
    return struct.pack('>4sBBBBIIHHIII', b'P6SR', 1, 1, rate, 0, h, w, lh, lw,
                       len(payloads[0]), len(payloads[1]), zlib.crc32(payload)) + payload


def independent_unpack(wire):
    assert isinstance(wire, bytes) and len(wire) >= 32
    magic, version, algorithm, rate, reserved, h, w, lh, lw, nl, nr, crc = struct.unpack('>4sBBBBIIHHIII', wire[:32])
    assert (magic, version, algorithm, reserved) == (b'P6SR', 1, 1, 0) and rate in (10, 30, 50)
    assert h > 0 and w > 0 and (lh, lw) == (math.floor(h / math.sqrt(rate)), math.floor(w / math.sqrt(rate)))
    assert nl == nr == lh * lw * 3 > 0 and len(wire) == 32 + nl + nr
    assert zlib.crc32(wire[32:]) == crc
    return [np.frombuffer(wire[32:32 + nl], dtype=np.uint8).reshape(lh, lw, 3).copy(),
            np.frombuffer(wire[32 + nl:], dtype=np.uint8).reshape(lh, lw, 3).copy()], (h, w), rate


def independent_receive(wire, values):
    low, (h, w), rate = independent_unpack(wire); outputs = []
    matrix = np.array([[65.481, 128.553, 24.966], [-37.797, -74.203, 112], [112, -93.786, -18.214]], dtype=np.float64)
    offset = np.array([16, 128, 128], dtype=np.float64); device = values['weight1'].device
    for image in low:
        channels = [np.asarray(Image.fromarray(image[:, :, k].astype(np.float32) / np.float32(255)).resize((w, h), Image.Resampling.BICUBIC), dtype=np.float64) for k in range(3)]
        colors = (np.stack(channels, -1) @ matrix.T + offset) / 255
        value = torch.from_numpy(colors[:, :, 0].astype(np.float32)[None, None].copy()).to(device)
        for index, radius in ((1, 4), (2, 2), (3, 2)):
            value = F.conv2d(F.pad(value, (radius,) * 4, mode='replicate'), values['weight' + str(index)], values['bias' + str(index)])
            if index < 3: value = value.clamp_min(0)
        colors[:, :, 0] = value[0, 0].detach().cpu().numpy().astype(np.float64)
        rgb = np.ascontiguousarray((colors * 255 - offset) @ np.linalg.inv(matrix).T, dtype='<f4')
        assert np.isfinite(rgb).all(); outputs.append(rgb)
    return outputs


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--scope', choices=('engineering', 'main'), required=True)
    parser.add_argument('--rate', type=int, choices=(10, 30, 50), required=True); args = parser.parse_args()
    source = c.CPU_gate(); runs, dependencies = c.gate(args.scope); free = c.cuda_gate()
    prefix = c.prefix(args.scope); encode_path = c.ROOT / 'data/runs' / (prefix + '-encode.json')
    manifest = c.ROOT / 'data/runs' / (prefix + '-cr' + str(args.rate) + '-receive.json')
    output = c.ROOT / 'data/provenance' / (prefix + '-cr' + str(args.rate) + '-audit.json'); assert not output.exists()
    parent, run = c.read(encode_path), c.read(manifest); ids = c.ids(args.scope)
    assert parent['sources'] == run['sources'] == source and parent['dependencies'] == run['dependencies'] == dependencies
    assert run['state'] == 'finished_all_native_received_pairs' and parent['state'] == 'finished_all_three_source_wire_conditions'
    assert run['ordered_ids'] == ids == parent['frame_ids'] and len(run['frames']) == len(ids)
    assert run['encoding_manifest_sha256'] == c.sha(encode_path)
    assert run['data_barrier'] and run['received_header_only_geometry'] and run['no_clean_GT_calibration_or_detector']
    assert not run['TF32'] and not run['mixed_precision'] and run['CPU_threads'] == 2 and run['cudnn_deterministic']
    assert run['model_initial_state'] == run['model_final_state']
    assert run['peak_reserved_bytes'] + 2 * 2**30 <= run['physical_free_before_bytes']
    selected = runs[args.rate]; assert run['checkpoint_sha256'] == selected['checkpoint_sha256']
    model = c.r.model_from_checkpoint(Path(selected['checkpoint']).read_bytes(), selected, source['training'],
                                     'engineering' if args.scope == 'engineering' else 'formal', args.rate, 'cuda')
    initial = c.r.states(model); assert initial == run['model_initial_state']
    values = {key: value.detach() for key, value in model.state_dict().items()}; torch.cuda.reset_peak_memory_stats()
    allowed = [c.DATA / 'training' / camera / (frame + '.png') for frame in ids for camera in ('image_2', 'image_3')]
    allowed += [Path(row[key]) for row in run['frames'] for key in ('wire_path', 'cache_path')]
    control = {'active': False}; sys.addaudithook(c.sensor_guard(allowed, control)); control['active'] = True
    rows = []; max_error = 0.; raw_total = wire_total = pixels = 0
    with torch.no_grad():
        for frame, encoded, received in zip(ids, parent['conditions'][str(args.rate)]['rows'], run['frames']):
            assert frame == encoded['frame_id'] == received['frame_id']
            assert encoded['wire_path'] == received['wire_path']
            left, l = c.png(c.DATA / 'training/image_2' / (frame + '.png')); right, r = c.png(c.DATA / 'training/image_3' / (frame + '.png'))
            assert l == encoded['source_left'] and r == encoded['source_right']
            wire = Path(encoded['wire_path']).read_bytes()
            assert c.r.sha_bytes(wire) == encoded['wire_sha256'] == received['wire_sha256']
            assert wire == independent_wire(left, right, args.rate)
            low, geometry, rate = independent_unpack(wire)
            assert received['native_hw'] == list(geometry) and received['low_hw'] == list(low[0].shape[:2])
            assert received['source_bytes'] == len(wire) and received['framing_bytes'] == 32 and received['nominal_compression'] == rate == args.rate
            assert received['received_header_only_geometry'] and received['output_unclipped']
            assert received['source_only_PHY_uses'] is None and received['source_only_PHY_energy'] is None
            expected = independent_receive(wire, values); blob = Path(received['cache_path']).read_bytes()
            tensors = c.r.tensor_pair(blob, received)
            pair_error = 0.
            with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
                for i, camera in enumerate(('left', 'right')):
                    actual = archive[camera]; difference = float(np.max(np.abs(actual.astype(np.float64) - expected[i].astype(np.float64))))
                    assert difference <= 3e-5 and actual.shape == left.shape
                    assert received['ranges'][i] == dict(minimum=float(actual.min()), maximum=float(actual.max()),
                                                       out_of_range_fraction=float(np.mean((actual < 0) | (actual > 1))))
                    assert np.array_equal(tensors[i][0].numpy().transpose(1, 2, 0), actual)
                    assert received['readonly_states'] == initial
                    pair_error = max(pair_error, difference); max_error = max(max_error, difference); pixels += actual.size
            meta = encoded['source_metadata']; assert meta['source_bytes'] == len(wire) and meta['framing_bytes'] == 32
            assert meta['raw_RGB8_bits'] == 8 * (left.size + right.size)
            raw_total += left.size + right.size; wire_total += len(wire)
            rows.append(dict(frame_id=frame, wire_sha256=received['wire_sha256'], cache_sha256=received['cache_sha256'],
                             arrays=received['arrays'], independent_max_abs_error=pair_error))
            assert torch.cuda.max_memory_reserved() + 2 * 2**30 <= free
    condition = parent['conditions'][str(args.rate)]
    assert raw_total == condition['raw_RGB8_bytes'] and wire_total == condition['wire_bytes']
    assert raw_total / wire_total == condition['actual_pooled_raw_to_wire_ratio']
    control['active'] = False
    assert c.r.states(model) == initial and c.sources() == source
    result = dict(state='passed_every_native_SRCNN_source_wire_float_pixel_and_readonly_state', checked_unix=time.time(),
                  scope=args.scope, rate=args.rate, pairs=len(ids), RGB_values=pixels, sources=source, dependencies=dependencies,
                  manifest_sha256=c.sha(manifest), encoding_manifest_sha256=c.sha(encode_path), maximum_RGB_error=max_error,
                  actual_pooled_raw_to_wire_ratio=raw_total / wire_total, wire_bytes=wire_total, rows=rows,
                  readonly_states=initial, physical_free_before_bytes=free, peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  limitation='Independent literal framing/color and direct full-array Torch convolutions; shared convolution backend, no local/GPU model backpropagation replay, quality or AP')
    with output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(state=result['state'], rate=args.rate, pairs=len(ids), RGB_values=pixels)), flush=True)


if __name__ == '__main__': main()
