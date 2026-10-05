"""Public received-byte, source, crop and closed-predecessor contracts."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROTOCOL = ROOT / 'experiments/original-paper-scenarios/ecsic-neural-reception-protocol-001.md'
LOCKED = {
 'experiments/original-paper-scenarios/ecsic-neural-reception-protocol-001.md': '61034c2453e6d80671d5fff63774777b260e74f2ce11ea2ffa95092415ec5c4a',
 'data/engineering/artemis-ecsic-digital-CPU-001.json': 'f265c56a7fdcc3e98fde0ff580758479aa50036d4b474fa949ec7cbc605227d3',
 'data/provenance/ecsic-received-transfer-index-001.json': 'd5026b31126e90035017890028b2e1daf4b12776c28e6971b8a90f755b6fc6de',
 'data/provenance/ecsic-digital-CPU-001-audit.json': '05d75ce7170697218776308d60b0d17b328e5d42f16379fed6b40a09135bf928',
 'data/provenance/artemis-ecsic-digital-CPU-001-terminal.json': 'c186a53680f529a6e384d73ae01f38af818d9299c7edf4402b5acd20df70f191',
 'data/provenance/artemis-ecsic-digital-audit-terminal-001.json': '7f7298d25d9c1ad737fcfcf42b0a6b240c9666e0b044d1f7b92d35a3b3cb4e03',
 'data/provenance/ecsic-digital-transferred-verification-001.json': '48ba48f1127bca9a631347ab4be2f9b123156580f2468421e8471f982ab3099e',
}
NATIVE = ROOT / 'experiments/original-paper-scenarios/ecsic-entropy-code/native.py'
CODEC = ROOT / 'experiments/original-paper-scenarios/ecsic-entropy-code/codec.py'
CODEC_SHA = 'a22fb7717cd1e2a13e4539a74a541958e93a7f6b215ae960b9c2736ad53ec157'
NATIVE_SHA = '33dbb471f341f73e8179ac100544c238141020ffdfe80909051add819da812b5'
CDF_SHA = '507d22e35fb0bb605f05a3312600a8aff3ff10f1ec1c4964358cb390e35d56e5'
STAGE_B = Path('/mnt/d/paper6/runs/ecsic-entropy-stageB-002')


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''): h.update(part)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())
def save(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def identities():
    old = read(ROOT / 'data/engineering/artemis-ecsic-digital-CPU-001.json')['sources_before']
    need(all(sha(ROOT / p) == h for p, h in old.items()), 'sealed historical sources changed')
    current = {str(p.relative_to(ROOT)): sha(p) for p in sorted(HERE.iterdir()) if p.is_file()}
    return dict(historical=old, current=current)


def codec():
    need(sha(CODEC) == CODEC_SHA and sha(NATIVE) == NATIVE_SHA, 'unchanged entropy decoder/parser')
    spec = importlib.util.spec_from_file_location('sealed_ecsic_neural_reception_codec', CODEC)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    _, digest = module.tables(); need(digest == CDF_SHA, 'fixed public CDF')
    return module


def predecessor():
    need(all(sha(ROOT / p) == h for p, h in LOCKED.items()), 'prelocked input evidence')
    manifest = read(ROOT / 'data/engineering/artemis-ecsic-digital-CPU-001.json')
    audit = read(ROOT / 'data/provenance/ecsic-digital-CPU-001-audit.json')
    need(audit['state'] == 'passed_all20_independent_actual_PHY_and_P6EC_frame_audits' and audit['actual_terminal'], 'independent full PHY terminal audit')
    need(audit['manifest_sha256'] == LOCKED['data/engineering/artemis-ecsic-digital-CPU-001.json'] and len(manifest['packets']) == 20, 'complete packet identity')
    index = read(ROOT / 'data/provenance/ecsic-received-transfer-index-001.json')
    proofs = {}
    for host, name in [('artemis', 'ecsic-received-artemis-fresh-001.json'), ('local', 'ecsic-received-local-transfer-001.json'), ('sheng', 'ecsic-received-sheng-transfer-001.json')]:
        p = ROOT / 'data/provenance' / name; value = read(p)
        need(value['state'] == 'passed_all22_actual_received_files' and value['host'] == host, 'fresh received-file transfer proof')
        need(value['files_sha256'] == index['files_sha256'] and value['index_sha256'] == sha(ROOT / 'data/provenance/ecsic-received-transfer-index-001.json'), 'all22 transferred bytes')
        proofs[name] = sha(p)
    need(len(index['files_sha256']) == 22 and all(sha(ROOT / p) == h for p, h in index['files_sha256'].items()), 'all actual received bytes on sheng')
    need(sum(x['reception']['state'] == 'received' for x in manifest['packets']) == 11 and sum(x['reception']['state'] == 'erased' for x in manifest['packets']) == 9, 'fixed20 outcomes')
    return manifest, dict(locked_evidence_sha256=LOCKED, transfer_proofs_sha256=proofs, received_files_sha256=index['files_sha256'])


def crop_arrays(left, right, original_hw, padded_hw):
    need(isinstance(original_hw, list) and isinstance(padded_hw, list) and len(original_hw) == len(padded_hw) == 2, 'public dimensions only')
    h, w = original_hw; ph, pw = padded_hw
    need(all(type(x) is int for x in (h, w, ph, pw)) and 0 < h <= ph <= 2048 and 0 < w <= pw <= 4096 and
         ph % 32 == pw % 32 == 0 and ph-h < 32 and pw-w < 32, 'locked top-left public padding')
    need(left.shape == right.shape == (1, 3, ph, pw) and left.dtype == right.dtype == np.float32, 'received FP32 NCHW RGB')
    need(np.isfinite(left).all() and np.isfinite(right).all(), 'finite received RGB')
    return dict(left=np.ascontiguousarray(left[:, :, :h, :w]), right=np.ascontiguousarray(right[:, :, :h, :w]))


def describe(value):
    value = np.ascontiguousarray(value)
    return dict(shape=list(value.shape), dtype=value.dtype.str, sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                finite=bool(np.isfinite(value).all()), min=float(value.min()), max=float(value.max()))


def input_guard(allowed_npz):
    allowed = {str(Path(p).resolve()) for p in allowed_npz}
    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)): return
        name = os.fsdecode(args[0]); mode, flags = args[1], args[2]
        reading = (isinstance(mode, str) and 'r' in mode) or (mode is None and flags & os.O_ACCMODE == os.O_RDONLY)
        if reading:
            need('/data/kitti/' not in name, 'crop tried clean KITTI input')
            if name.endswith('.npz'): need(str(Path(name).resolve()) in allowed, 'crop tried foreign/source NPZ')
    return guard
