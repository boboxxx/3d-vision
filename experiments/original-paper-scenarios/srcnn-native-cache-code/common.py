"""Native SRCNN cache identities, complete training gates and received routing."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FLOAT = HERE.parent / 'srcnn-float-reception-code'
sys.path.insert(0, str(FLOAT))
import receiver as r
_import_paths = list(sys.path)
try:
    sys.path.insert(0, str(HERE.parent / 'srcnn-KITTI-adaptation-code-002'))
    train = r.module('native_cache_training', HERE.parent / 'srcnn-KITTI-adaptation-code-002/common.py')
finally:
    sys.path[:] = _import_paths
DATA = Path('/mnt/d/paper6/data/kitti')
NATIVE = Path('/mnt/d/paper6/runs')
VAL_SHA = '657ac4bcc1e156e5b106a4ca18e1f88e012787ea1d2b5d0adeea97fee903fa86'


def sha(path): return train.sha(path)
def read(path): return train.read(path)
def save(path, value): return train.save(path, value)


def sources():
    files = list(HERE.glob('*.py')); assert len(files) == 5
    files += list(FLOAT.glob('*.py'))
    files += [HERE.parent / 'srcnn-native-cache-audit-001.md', r.PROTOCOL]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in files}
    assert sha(r.PROTOCOL) == r.PROTOCOL_SHA
    return dict(cache=hashes, training=train.sources())


def gate(scope):
    assert scope in ('engineering', 'main')
    training_scope = 'engineering' if scope == 'engineering' else 'formal'
    prefix = 'srcnn-kitti-' + training_scope + '-seed17-002'
    closure_path = ROOT / 'data/provenance' / (prefix + '-closure.json')
    closure = read(closure_path); proof = read(ROOT / 'data/provenance' / (prefix + '-local-verification.json'))
    expected = 18 if scope == 'engineering' else 111360
    assert closure['state'] == 'closed_actual_terminal_all_three_rates_and_paired_training'
    assert closure['actual_terminal'] and closure['scope'] == training_scope and closure['updates'] == expected
    assert closure['sources'] == proof['sources'] == train.sources()
    assert proof['state'] == 'passed_all' + str(expected) + '_transferred_SRCNN_training_records_and_terminal_artifacts'
    assert proof['closure_sha256'] == sha(closure_path)
    if scope == 'main':
        assert proof['updates'] == 111360 and proof['source_views'] == 7424 and proof['artifacts_verified'] == 21
        assert proof['independently_regenerated_view_and_patch_streams']
        assert proof['verifier_sha256'] == sha(ROOT / 'data/provenance/verify-srcnn-kitti-complete-local-003.py')
        assert proof['acceptance_plan_sha256'] == sha(HERE.parent / 'srcnn-KITTI-formal-record-validation-001.md')
    float_proof = read(ROOT / 'data/provenance/srcnn-float-reception-transferred-verification-001.json')
    assert float_proof['state'] == 'passed_all12_transferred_trained_float_reception_view_records'
    assert float_proof['training_sources'] == train.sources() and float_proof['protocol_sha256'] == r.PROTOCOL_SHA
    for path, digest in float_proof['sources'].items(): assert sha(ROOT / path) == digest
    for path, digest in float_proof['reports'].items():
        assert sha(ROOT / path) == digest
        report = read(ROOT / path)
        assert report['state'] == 'passed_six_trained_SRCNN_float_reception_CPU_families' and report['families'] == 6
    runs = {}
    cycle = read(ROOT / 'data/runs' / (prefix + '-cycle.json'))
    assert len(cycle['completed_commands']) == 6 and all(v['returncode'] == 0 for v in cycle['completed_commands'])
    assert set(cycle['rates']) == {'10', '30', '50'}
    for rate in r.interface.RATES:
        item = cycle['rates'][str(rate)]; run = read(item['manifest']); report = read(item['audit'])
        assert sha(item['manifest']) == item['manifest_sha256'] == report['manifest_sha256']
        assert sha(item['audit']) == item['audit_sha256']
        assert run['state'] == 'finished' and report['state'] == 'passed_all_updates_native_independent_patches_and_final_Adam'
        assert run['sources'] == report['sources'] == train.sources()
        assert run['scope'] == training_scope and run['rate'] == rate
        assert run['optimizer_steps'] == report['updates'] == expected // 3
        runs[rate] = run
    dependencies = dict(training_closure_sha256=sha(closure_path), training_local_proof_sha256=sha(ROOT / 'data/provenance' / (prefix + '-local-verification.json')),
                        float_CPU_local_proof_sha256=sha(ROOT / 'data/provenance/srcnn-float-reception-transferred-verification-001.json'))
    return runs, dependencies


def ids(scope):
    if scope == 'engineering': return ['000000', '000003']
    path = DATA / 'ImageSets/val.txt'; assert sha(path) == VAL_SHA
    values = path.read_text().splitlines(); assert len(values) == len(set(values)) == 3769
    return values


def prefix(scope): return 'srcnn-source-cache-' + scope + '-001'


def sensor_guard(paths, control):
    allowed = {str(Path(p).resolve()) for p in paths}
    def barrier(event, values):
        if event != 'open' or not isinstance(values[0], (str, bytes)) or not control['active']: return
        name = os.fsdecode(values[0]); mode, flags = values[1:3]
        reading = (isinstance(mode, str) and ('r' in mode or '+' in mode)) or (mode is None and flags & os.O_ACCMODE != os.O_WRONLY)
        if not reading: return
        assert '/label_2/' not in name and '/calib/' not in name and '/velodyne/' not in name
        if name.lower().endswith(('.png', '.p6sr', '.npz', '.pkl', '.pth', '.pt', '.mat', '.bin')):
            assert str(Path(name).resolve()) in allowed, 'foreign source/cache/model read forbidden'
    return barrier


def png(path):
    blob = Path(path).read_bytes()
    with Image.open(io.BytesIO(blob)) as image:
        assert image.mode == 'RGB'; value = np.array(image, dtype=np.uint8)
    return value, dict(path=str(path), PNG_sha256=r.sha_bytes(blob), RGB8=r.describe(value))


def cuda_gate():
    assert ROOT == Path('/home/sheng/paper6') and os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    for name in ('source-cache-inference-main-001', 'srcnn-kitti-formal-seed17-002'):
        path = ROOT / 'data/runs' / (name + '-launch.json')
        if path.exists():
            status = subprocess.run(['ps', '-p', str(read(path)['pid'])], capture_output=True)
            assert status.returncode == 1, 'Existing main inference/formal training still live'
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    assert not active.strip(), 'Competing GPU compute process'
    free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip()) * 2**20
    assert free >= 12 * 2**30
    torch.set_num_threads(2); torch.manual_seed(17); torch.cuda.manual_seed_all(17)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    return free


def CPU_gate():
    source = sources()
    for host in ('local', 'sheng'):
        report = read(ROOT / 'data/engineering' / ('srcnn-native-cache-' + host + '-CPU-002.json'))
        assert report['state'] == 'passed_five_native_SRCNN_cache_CPU_families' and report['families'] == 5
        assert report['sources'] == source and report['views'] == 6 and report['maximum_RGB_error'] <= 3e-5
        assert report['actual_file_open_barriers_checked'] and report['guaranteed_overshoot_cache_views'] == 2
        assert [row['rate'] for row in report['records']] == [10, 30, 50]
    return source
