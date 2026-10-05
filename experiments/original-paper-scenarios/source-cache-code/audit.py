"""Independent native framing/direct-Pillow/all-pixel/source audit."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import platform
import struct
import time
import zlib

import numpy as np
import PIL
from PIL import Image, features

ROOT = Path(__file__).resolve().parents[3]
CONDITIONS = [(f'{codec}-cr{rate}', codec, rate, {10: 90, 30: 39, 50: 17}[rate] if codec == 'jpeg' else rate)
              for codec in ('jpeg', 'jpeg2000') for rate in (10, 30, 50)]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def describe(a):
    a = np.ascontiguousarray(a); descriptor = dict(dtype=a.dtype.str, shape=list(a.shape))
    return dict(**descriptor, sha256=hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode() + b'\0' + a.tobytes()).hexdigest())


def verify_pair(job):
    source, received, key, codec, rate, parameter = job
    assert source['frame_id'] == received['frame_id'] and source['condition'] == key and source['codec'] == codec
    assert source['nominal_compression'] == rate and source['parameter'] == parameter
    frame = source['frame_id']; evidence = source['evidence']
    expected_images = {str(Path('/mnt/d/paper6/data/kitti/training') / view / (frame + '.png')) for view in ('image_2', 'image_3')}
    assert set(source['source_images_sha256']) == expected_images
    native = []
    for view in ('image_2', 'image_3'):
        path = Path('/mnt/d/paper6/data/kitti/training') / view / (frame + '.png')
        assert sha(path) == source['source_images_sha256'][str(path)]
        with Image.open(path) as image:
            assert image.mode == 'RGB'; native.append(np.array(image, dtype=np.uint8))
    assert describe(native[0]) == evidence['left_input'] and describe(native[1]) == evidence['right_input']
    wire_path = Path(source['wire_path']); wire = wire_path.read_bytes()
    assert received['wire_path'] == source['wire_path'] and sha(wire_path) == evidence['wire_sha256'] == received['wire_sha256']
    magic, version, code, reserved, nl, nr, crc = struct.unpack_from('>4sBBHIII', wire)
    assert (magic, version, code, reserved) == (b'P6SB', 1, 1 if codec == 'jpeg' else 2, 0)
    assert nl > 0 and nr > 0 and len(wire) == 20 + nl + nr and zlib.crc32(wire[20:]) == crc
    assert evidence['codec'] == codec and evidence['parameter'] == parameter and evidence['codestream_bytes'] == [nl, nr]
    expected_options = (dict(quality=parameter, subsampling=2, optimize=False, progressive=False) if codec == 'jpeg' else
                        dict(quality_mode='rates', quality_layers=[parameter], irreversible=True, mct=1, no_jp2=False))
    assert evidence['options'] == expected_options and evidence['framing_bytes'] == 20
    assert evidence['source_bytes'] == len(wire) and evidence['source_bits'] == len(wire) * 8
    actual = []
    for blob in (wire[20:20 + nl], wire[20 + nl:]):
        with Image.open(io.BytesIO(blob)) as image:
            assert image.format == ('JPEG' if codec == 'jpeg' else 'JPEG2000') and image.mode == 'RGB'
            actual.append(np.array(image, dtype=np.uint8))
    assert native[0].shape == native[1].shape == actual[0].shape == actual[1].shape
    assert describe(actual[0]) == evidence['left_received'] == received['arrays']['left']
    assert describe(actual[1]) == evidence['right_received'] == received['arrays']['right']
    assert received['public_hw'] == list(actual[0].shape[:2])
    cache_path = Path(received['cache_path']); assert sha(cache_path) == received['cache_sha256']
    with np.load(cache_path, allow_pickle=False) as pixels:
        assert set(pixels.files) == {'left', 'right'}
        for name, expected in zip(('left', 'right'), actual):
            assert pixels[name].dtype == np.uint8 and np.array_equal(pixels[name], expected)
    raw_bytes = native[0].size + native[1].size
    assert evidence['raw_RGB8_bits'] == raw_bytes * 8 and evidence['actual_raw_to_serialized_ratio'] == raw_bytes / len(wire)
    return dict(frame_id=frame, raw_bytes=raw_bytes, wire_bytes=len(wire), pixels_exact=actual[0].size + actual[1].size,
                native_files={str(wire_path): sha(wire_path), str(cache_path): sha(cache_path)})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--scope', choices=('engineering', 'main'), required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    assert not args.output.exists()
    directory = args.directory.resolve(); encoding_path = directory / 'encode.json'; encoding = read(encoding_path)
    assert encoding['state'] == 'finished_all_six_complete_source_wires' and encoding['scope'] == args.scope
    environment = dict(Python=platform.python_version(), NumPy=np.__version__, Pillow=PIL.__version__,
        libjpeg=features.version('jpg'), libjpeg_turbo=features.version('libjpeg_turbo'), OpenJPEG=features.version('jpg_2000'))
    assert encoding['environment'] == environment == dict(Python='3.10.15', NumPy='1.26.3', Pillow='10.2.0', libjpeg='6.2', libjpeg_turbo='3.0.1', OpenJPEG='2.5.0')
    for group in encoding['sources'].values():
        for p, digest in group.items(): assert sha(ROOT / p) == digest
    for p, digest in encoding['predecessor'].items(): assert sha(ROOT / p) == digest
    assert sha(ROOT / 'experiments/original-paper-scenarios/source-cache-protocol-001.md') == encoding['protocol_sha256'] == '84ec93430f9fdea3d1aacaaf6453cda81a8bfa5e247ed1cc39e535807e7d098e'
    if args.scope == 'main':
        split = Path('/mnt/d/paper6/data/kitti/ImageSets/val.txt')
        assert sha(split) == '657ac4bcc1e156e5b106a4ca18e1f88e012787ea1d2b5d0adeea97fee903fa86'; ids = split.read_text().splitlines()
        assert len(ids) == 3769
    else: ids = ['000000', '000003']
    assert encoding['frame_ids'] == ids and encoding['actual_encodes'] == 6 * len(ids)
    assert set(encoding['conditions']) == {x[0] for x in CONDITIONS} and encoding['input_barrier'] and encoding['no_GT_model_AP_or_radio']
    assert encoding['source_only_PHY_uses'] is None and encoding['source_only_PHY_energy'] is None
    native_files = {str(encoding_path): sha(encoding_path)}; summaries = {}; receiver_pids = set(); pairs = pixels = 0
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
        for key, codec, rate, parameter in CONDITIONS:
            receiver_path = directory / 'received' / key / 'receiver.json'; receiver = read(receiver_path)
            assert receiver['state'] == 'finished_all_received_pairs' and receiver['scope'] == args.scope and receiver['condition'] == key
            assert receiver['environment'] == environment and receiver['sources'] == encoding['sources'] and receiver['protocol_sha256'] == encoding['protocol_sha256']
            assert receiver['input_barrier'] and not receiver['PNG_or_source_manifest_inputs'] and receiver['dimensions_from_received_stream_headers'] and receiver['HWC_uint8_no_resize'] and receiver['no_model_AP_or_radio']
            assert receiver['frame_ids'] == ids and receiver['pid'] not in receiver_pids and receiver['pid'] != encoding['pid']; receiver_pids.add(receiver['pid'])
            source_records = encoding['conditions'][key]['records']; received_records = receiver['records']
            assert [x['frame_id'] for x in source_records] == [x['frame_id'] for x in received_records] == ids
            assert {p.name for p in (directory / 'source' / key).iterdir()} == {x + '.p6sb' for x in ids}
            assert {p.name for p in (directory / 'received' / key).iterdir()} == {'receiver.json', *(x + '.npz' for x in ids)}
            raw_total = wire_total = pixel_total = 0
            for record in pool.map(verify_pair, [(s, r, key, codec, rate, parameter) for s, r in zip(source_records, received_records)], chunksize=1):
                pairs += 1; raw_total += record['raw_bytes']; wire_total += record['wire_bytes']; pixel_total += record['pixels_exact']
                native_files.update(record['native_files'])
            pixels += pixel_total; native_files[str(receiver_path)] = sha(receiver_path)
            summaries[key] = dict(pairs=len(ids), full_pixel_values_exact=pixel_total, raw_RGB8_bytes=raw_total,
                full_wire_bytes=wire_total, pooled_raw_to_wire_ratio=raw_total / wire_total)
    assert pairs == 6 * len(ids) and len(receiver_pids) == 6 and len(native_files) == 12 * len(ids) + 7
    result = dict(state='passed_all_six_complete_conditions_full_native_pixels_and_sources', scope=args.scope, checked_unix=time.time(),
        protocol_sha256=encoding['protocol_sha256'], encoding_manifest_sha256=sha(encoding_path), pairs=pairs, views=2 * pairs,
        full_pixel_values_exact=pixels, six_distinct_receiver_processes=True, conditions=summaries,
        native_files_sha256=native_files, sources=encoding['sources'], no_GT_model_AP_or_radio=True,
        scope_note='Complete native source/wire/cache pixels audited on sheng; not detector AP or local raw pixel replay')
    with args.output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(state=result['state'], pairs=pairs, views=2 * pairs)), flush=True)


if __name__ == '__main__': main()
