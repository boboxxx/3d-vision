"""Fixed training-only real JPEG byte calibration, no detector or radio."""
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import PIL
from PIL import Image, features

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / 'code'))
from image_codecs import encode_pair
PREFIX = 'original-jpeg-rate-calibration-001'
DATA = Path('/mnt/d/paper6/data/kitti')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def save(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def environment():
    return dict(Python=platform.python_version(), NumPy=np.__version__, Pillow=PIL.__version__,
                libjpeg=features.version('jpg'), libjpeg_turbo=features.version('libjpeg_turbo'),
                OpenJPEG=features.version('jpg_2000'))


def main():
    output = ROOT / 'data/runs' / (PREFIX + '.json')
    directory = Path('/mnt/d/paper6/runs') / PREFIX
    assert not output.exists() and not directory.exists() and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    expected_environment = dict(Python='3.10.15', NumPy='1.26.3', Pillow='10.2.0', libjpeg='6.2', libjpeg_turbo='3.0.1', OpenJPEG='2.5.0')
    assert environment() == expected_environment
    protocol = HERE.parent / 'jpeg-rate-calibration-protocol-001.md'
    fold_path = ROOT / 'data/internal-tuning-fold-001.json'
    gate_path = ROOT / 'data/engineering/original-scenarios-sheng-CPU-001.json'
    assert sha(protocol) == '2f1f514cbcd6b2005d97b292478953875cdbd40c025dac42267b4998844db955'
    assert sha(fold_path) == '35f22b0022b5968c09c9a8a7e1b9ad6c8ab23c52283673495a36ddba59406f4c'
    assert sha(gate_path) == '64cadae456fcc2e0d94bc3c972d5e3a93275ae02acb0af20ca03969afc6ccb46'
    gate = read(gate_path)
    assert gate['state'] == 'passed' and gate['tests_passed'] == 7
    historical = {str((HERE.parent / 'code' / p).relative_to(ROOT)): h for p, h in gate['source_files'].items()}
    assert all(sha(ROOT / p) == h for p, h in historical.items())
    sources = dict(historical=historical, current={str(p.relative_to(ROOT)): sha(p) for p in sorted(HERE.iterdir()) if p.is_file()})
    fold = read(fold_path); ids = sorted(fold['folds']['geocomm_tune_train']['ids'])[:64]
    assert len(ids) == len(set(ids)) == 64 and set(ids).isdisjoint(fold['folds']['geocomm_tune_holdout']['ids'])
    allowed = {str((DATA / 'training' / view / (frame + '.png')).resolve()) for frame in ids for view in ('image_2', 'image_3')}
    def barrier(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)): return
        name = os.fsdecode(args[0]); mode, flags = args[1:3]
        reading = (isinstance(mode, str) and 'r' in mode) or (mode is None and flags & os.O_ACCMODE == os.O_RDONLY)
        if reading:
            assert not name.endswith(('.pkl', '.pth', '.pt', '.npz')), 'model/infos/NPZ input forbidden'
            if '/data/kitti/' in name: assert str(Path(name).resolve()) in allowed, 'noncalibration KITTI input forbidden'
    sys.addaudithook(barrier)
    directory.mkdir(parents=True)
    raw = directory / 'calibration.jsonl'
    info = dict(state='running', prefix=PREFIX, pid=os.getpid(), started_unix=time.time(), directory=str(directory),
                protocol_sha256=sha(protocol), fold_sha256=sha(fold_path), CPU_gate_sha256=sha(gate_path),
                sources=sources, environment=environment(), frame_ids=ids, completed_qualities=0, actual_records=0,
                source_images_before={}, no_GT_model_mainval_or_radio=True, input_barrier=True,
                candidate_wires_retained=False, pooled_ratios={}, selected_qualities={})
    save(output, info)
    try:
        images = {}
        for frame in ids:
            paths = [DATA / 'training' / view / (frame + '.png') for view in ('image_2', 'image_3')]
            pair = []
            for p in paths:
                info['source_images_before'][str(p)] = sha(p)
                with Image.open(p) as image:
                    assert image.mode == 'RGB'; pair.append(np.array(image, dtype=np.uint8))
            assert pair[0].shape == pair[1].shape
            images[frame] = pair
        save(output, info)
        with raw.open('x', buffering=1) as stream:
            for quality in range(1, 96):
                raw_bytes = wire_bytes = 0
                for frame in ids:
                    wire, decoded, record = encode_pair(*images[frame], codec='jpeg', parameter=quality)
                    assert len(wire) == record['source_bytes'] and all(x.shape == images[frame][0].shape for x in decoded)
                    record.update(frame_id=frame, quality=quality)
                    stream.write(json.dumps(record, allow_nan=False, separators=(',', ':')) + '\n')
                    raw_bytes += record['raw_RGB8_bits'] // 8; wire_bytes += len(wire); info['actual_records'] += 1
                ratio = raw_bytes / wire_bytes
                info['pooled_ratios'][str(quality)] = dict(raw_RGB8_bytes=raw_bytes, full_wire_bytes=wire_bytes, ratio=ratio)
                info['completed_qualities'] = quality; save(output, info)
                print(json.dumps(dict(quality=quality, ratio=ratio, records=info['actual_records'])), flush=True)
        for target in (10, 30, 50):
            quality = min(range(1, 96), key=lambda q: (abs(math.log(info['pooled_ratios'][str(q)]['ratio'] / target)), q))
            ratio = info['pooled_ratios'][str(quality)]['ratio']
            info['selected_qualities'][str(target)] = dict(quality=quality, measured_ratio=ratio, absolute_log_residual=abs(math.log(ratio / target)))
        assert info['actual_records'] == 6080
        assert all(sha(p) == h for p, h in info['source_images_before'].items())
        assert all(sha(ROOT / p) == h for group in sources.values() for p, h in group.items())
        assert sha(protocol) == info['protocol_sha256'] and sha(fold_path) == info['fold_sha256'] and environment() == info['environment']
        info.update(state='finished6080_real_pair_encodes_independent_audit_pending', records_path=str(raw), records_sha256=sha(raw),
                    source_images_after=info['source_images_before'], ended_unix=time.time())
    except BaseException as error:
        info.update(state='failed', exception=repr(error), ended_unix=time.time()); raise
    finally: save(output, info)
    print(json.dumps(dict(state=info['state'], selected=info['selected_qualities'])), flush=True)


if __name__ == '__main__': main()
