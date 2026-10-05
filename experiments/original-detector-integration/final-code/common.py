"""Shared identities and strict received-cache contract, no detector inference."""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from geocomm.evidence import sha256, source_identity

ORIGINAL_PROTOCOL_SHA = '682fb774e09b9be7f68d09f8251dac6d4b99af3290f02660951d398f9e00eace'
FINAL_PROTOCOL_SHA = '5e24dc53641749f035f2e2f4d354505cd286511bededa3ad0a3993bf168592d4'
FOLD_SHA = '35f22b0022b5968c09c9a8a7e1b9ad6c8ab23c52283673495a36ddba59406f4c'
EPOCHS = (12, 10, 6, 10, 45)
NATIVE_PROBES = {
    'stereo_rcnn': 'original-final-StereoRCNN-receiver-engineering-001.json',
    'liga': 'original-final-LIGA-receiver-engineering-001.json',
}


def check(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def tensor_identity(value):
    check(isinstance(value, torch.Tensor) and value.device.type == 'cpu', 'CPU tensor required')
    check(value.dtype == torch.float32 and value.ndim == 4 and value.shape[:2] == (1, 3)
          and torch.isfinite(value).all(), 'finite native float32 RGB required')
    shape = list(value.shape)
    digest = hashlib.sha256()
    digest.update(json.dumps(dict(dtype='float32', shape=shape), sort_keys=True).encode())
    digest.update(value.detach().contiguous().numpy().tobytes())
    return dict(shape=shape, dtype='float32', array_sha256=digest.hexdigest(),
                minimum=float(value.min()), maximum=float(value.max()),
                out_of_range_fraction=float(((value < 0) | (value > 1)).float().mean()))


def noise_identity(generator):
    return hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()


def ordered_ids():
    path = ROOT / 'data/internal-tuning-fold-001.json'
    check(sha256(path) == FOLD_SHA, 'fixed fold changed')
    ids = read(path)['folds']['geocomm_tune_holdout']['ids']
    check(len(ids) == len(set(ids)) == 372, 'full ordered372 fold')
    return ids


def source_specs():
    return dict(project=(ROOT, ['src', 'scripts', 'configs', 'pyproject.toml']),
                liga=(ROOT / 'third_party/LIGA-Stereo', ['liga', 'configs', 'tools', 'setup.py']),
                mmdet=(ROOT / 'third_party/mmdetection_kitti', ['mmdet']),
                stereo_rcnn=(ROOT / 'third_party/Stereo-RCNN', ['lib', 'demo.py', 'test_net.py']),
                final_evaluation=(ROOT, ['experiments/original-detector-integration/final-code',
                                         'experiments/original-detector-integration/adapter.py']))


def current_sources():
    check(sha256(HERE / 'evaluation-protocol.md') == FINAL_PROTOCOL_SHA, 'final evaluation protocol changed')
    original = read(ROOT / 'data/provenance/cao2025-staged-trainer-source-bfee9d8.json')
    check(len(original) == 21 and all(sha256(ROOT / 'reproduction/cao2025' / k) == v
                                    for k, v in original.items()), 'original21 source changed')
    check(sha256(ROOT / 'reproduction/cao2025/formal-protocol.md') == ORIGINAL_PROTOCOL_SHA,
          'original protocol changed')
    return {k: source_identity(*v) for k, v in source_specs().items()}


def terminal(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def final_original_chain():
    """Final sole weight after full chain audit; no partial/best checkpoint."""
    check(terminal(26642), 'original full cycle still live')
    sources = current_sources()
    stages = []
    previous = None
    original = read(ROOT / 'data/provenance/cao2025-staged-trainer-source-bfee9d8.json')
    for stage, epochs in enumerate(EPOCHS, 1):
        path = ROOT / f'data/runs/cao2025-full-native-seed17-001-stage{stage}.json'
        audit_path = path.with_name(path.stem + '-audit.json')
        run, audit = read(path), read(audit_path)
        check(run['state'] == 'finished' and audit['state'] == 'passed' and run['stage'] == stage,
              'full audited stage required')
        check(run['source_files'] == audit['source_files'] == original and
              run['protocol_sha256'] == ORIGINAL_PROTOCOL_SHA and run['fold_sha256'] == FOLD_SHA,
              'stage source/protocol/fold')
        check(audit['manifest_sha256'] == sha256(path) and len(run['epochs']) == len(audit['epochs']) == epochs,
              'complete stage audit identity')
        check(run['evidence_type'] == 'exploratory_full_native_original_variant'
              and run['samples'] == audit['samples'] == 3340 * epochs, 'formal stage attempt budget')
        final = run['epochs'][-1]
        check(final['epoch'] == epochs and sha256(final['checkpoint_path']) == final['checkpoint_sha256']
              == audit['checkpoint_sha256'], 'final stage full checkpoint identity')
        if previous is not None:
            check(run['predecessor_manifest_sha256'] == previous['manifest_sha256']
                  and run['predecessor_audit_sha256'] == previous['audit_sha256']
                  and run['predecessor_checkpoint_sha256'] == previous['checkpoint_sha256'], 'whole parent chain')
        previous = dict(stage=stage, manifest_sha256=sha256(path), audit_sha256=sha256(audit_path),
                        checkpoint_path=final['checkpoint_path'], checkpoint_sha256=final['checkpoint_sha256'])
        stages.append(previous)
    queue_path = ROOT / 'data/runs/native-validation-queue-002.json'
    queue = read(queue_path)
    check(queue['state'] == 'finished_all_native_engineering_closed' and terminal(queue['pid']),
          'native validation queue must be fully closed and terminal')
    probes = {}
    for detector, name in NATIVE_PROBES.items():
        path = ROOT / 'data/engineering' / name
        probe = read(path)
        count = 484 if detector == 'liga' else 670
        check(probe['state'] == 'passed' and probe['detector_states_readonly'] == count
              and probe['codec_states_readonly'] == 768 and probe['source_identities'] == sources
              and [r['channel'] for r in probe['conditions']] == ['clean_relay', 'identity', 'awgn'],
              'new receiver/cache native engineering required')
        probes[detector] = probe
    for a, b in zip(probes['stereo_rcnn']['conditions'], probes['liga']['conditions']):
        check(all(a[k] == b[k] for k in ('received_tensors', 'accounting', 'erasure')), 'paired engineering received inputs')
    return dict(stages=stages, checkpoint_path=previous['checkpoint_path'],
                checkpoint_sha256=previous['checkpoint_sha256'], sources=sources,
                native_queue_sha256=sha256(queue_path),
                native_probe_sha256={k: sha256(ROOT / 'data/engineering' / v) for k, v in NATIVE_PROBES.items()})


def load_received(cache_path, frame_id):
    """Strict received image pair only; no clean-image substitution on erasure."""
    cache_path = Path(cache_path)
    manifest = read(cache_path)
    check(manifest['state'] == 'finished' and manifest['channel'] in ('identity', 'awgn')
          and manifest['ordered_ids'] == ordered_ids() and manifest['frames_complete'] == 372,
          'fully sealed received cache required')
    row = manifest['frames'][frame_id]
    path = Path(row['received_path'])
    check(sha256(path) == row['received_file_sha256'], 'received file changed')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    check(set(payload) == {'frame_id', 'channel', 'outputs'} and payload['frame_id'] == frame_id
          and payload['channel'] == manifest['channel'], 'received-only cache envelope')
    outputs = payload['outputs']
    if row['erasure'] is not None:
        check(outputs is None and row['received_tensors'] is None, 'erasure cannot have fallback tensors')
    else:
        check(isinstance(outputs, tuple) and len(outputs) == 2
              and [tensor_identity(v) for v in outputs] == row['received_tensors'], 'received array identity')
        check(outputs[0].shape == outputs[1].shape == tuple(row['native_shape']), 'native stereo cache shape')
    return outputs, row
