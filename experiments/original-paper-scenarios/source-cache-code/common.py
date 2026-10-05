"""Pinned native source-cache context, public conditions and input barriers."""
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np
import PIL
from PIL import features

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = Path('/mnt/d/paper6/data/kitti')
PROTOCOL = HERE.parent / 'source-cache-protocol-001.md'
PROTOCOL_SHA = '84ec93430f9fdea3d1aacaaf6453cda81a8bfa5e247ed1cc39e535807e7d098e'
VAL_SHA = '657ac4bcc1e156e5b106a4ca18e1f88e012787ea1d2b5d0adeea97fee903fa86'
LOCKED = {
 'data/runs/original-jpeg-rate-calibration-001.json': 'ad41d390c7084a7279920cce7d881f327071630b3bf8852c65dbc88033379d62',
 'data/provenance/original-jpeg-rate-calibration-001-audit.json': 'e014bc8b45d172189eaaa43b7aeca7abb1cbdd1a34401fe59aa5ecca28dc5ecd',
 'data/provenance/original-jpeg-rate-calibration-001-local-verification-001.json': '320570301851fbabd5024ee70253839c7bcdcad517b06e06248a326f773dcceb',
 'data/engineering/original-scenarios-sheng-CPU-001.json': '64cadae456fcc2e0d94bc3c972d5e3a93275ae02acb0af20ca03969afc6ccb46',
}
ENVIRONMENT = dict(Python='3.10.15', NumPy='1.26.3', Pillow='10.2.0', libjpeg='6.2', libjpeg_turbo='3.0.1', OpenJPEG='2.5.0')
CONDITIONS = [(f'{codec}-cr{rate}', codec, rate, {10: 90, 30: 39, 50: 17}[rate] if codec == 'jpeg' else rate)
              for codec in ('jpeg', 'jpeg2000') for rate in (10, 30, 50)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1048576), b''): h.update(b)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())
def save(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def environment():
    return dict(Python=platform.python_version(), NumPy=np.__version__, Pillow=PIL.__version__,
                libjpeg=features.version('jpg'), libjpeg_turbo=features.version('libjpeg_turbo'), OpenJPEG=features.version('jpg_2000'))


def native_environment():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and environment() == ENVIRONMENT


def frames(scope):
    assert scope in ('engineering', 'main')
    if scope == 'main':
        p = DATA / 'ImageSets/val.txt'; assert sha(p) == VAL_SHA
        ids = p.read_text().splitlines(); assert len(ids) == len(set(ids)) == 3769
    else: ids = ['000000', '000003']
    assert all(len(x) == 6 and x.isdecimal() for x in ids)
    return ids


def identities():
    assert sha(PROTOCOL) == PROTOCOL_SHA
    old = read(ROOT / 'data/engineering/original-scenarios-sheng-CPU-001.json')['source_files']
    historical = {str((HERE.parent / 'code' / p).relative_to(ROOT)): h for p, h in old.items()}
    assert all(sha(ROOT / p) == h for p, h in historical.items())
    current = {str(p.relative_to(ROOT)): sha(p) for p in sorted(HERE.glob('*.py'))}
    assert len(current) == 5
    return dict(historical=historical, current=current)


def predecessor():
    assert all(sha(ROOT / p) == h for p, h in LOCKED.items())
    info = read(ROOT / 'data/runs/original-jpeg-rate-calibration-001.json')
    audit = read(ROOT / 'data/provenance/original-jpeg-rate-calibration-001-audit.json')
    local = read(ROOT / 'data/provenance/original-jpeg-rate-calibration-001-local-verification-001.json')
    assert audit['actual_terminal'] and audit['rows'] == local['rows'] == 6080 and audit['manifest_sha256'] == LOCKED['data/runs/original-jpeg-rate-calibration-001.json']
    assert local['state'] == 'passed_all6080_transferred_byte_records_and_rate_choices'
    assert {int(k): v['quality'] for k, v in info['selected_qualities'].items()} == {10: 90, 30: 39, 50: 17}
    gate = read(ROOT / 'data/engineering/original-scenarios-sheng-CPU-001.json')
    assert gate['state'] == 'passed' and gate['tests_passed'] == 7
    fold = read(ROOT / 'data/internal-tuning-fold-001.json')
    assert {'000000', '000003'} <= set(fold['folds']['geocomm_tune_train']['ids'])
    return LOCKED


def guard(role, allowed_png=()):
    assert role in ('source', 'receiver')
    allowed = {str(Path(p).resolve()) for p in allowed_png}
    public = str((DATA / 'ImageSets/val.txt').resolve())
    def barrier(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)): return
        name = os.fsdecode(args[0]); mode, flags = args[1:3]
        reading = (isinstance(mode, str) and 'r' in mode) or (mode is None and flags & os.O_ACCMODE == os.O_RDONLY)
        if not reading: return
        assert not name.endswith(('.pkl', '.pth', '.pt', '.npz')), 'model/infos/foreignNPZ read forbidden'
        if name.lower().endswith('.png'):
            assert role == 'source' and str(Path(name).resolve()) in allowed, 'receiver or foreign PNG read forbidden'
        if '/data/kitti/' in name:
            assert str(Path(name).resolve()) in allowed | {public}, 'GT/calibration/nonpublic KITTI input forbidden'
    return barrier


def hwc(value):
    assert value.dtype == np.uint8 and value.ndim == 3 and value.shape[2] == 3 and all(x > 0 for x in value.shape)
    return value
