"""Pinned original model on allocated PRO6000; no formal research training."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[2]
    original = root / 'reproduction/cao2025'
    assets = root / 'assets/original-native-001'
    output = root / 'data/engineering/artemis-original-native-001.json'
    native = output.with_name('artemis-original-native-001-phases.json')
    if output.exists() or native.exists() or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('unique evidence and actual Slurm allocation required')
    expected = json.loads((root / 'data/provenance/cao2025-staged-trainer-source-bfee9d8.json').read_text())
    if len(expected) != 21 or any(sha(original / k) != v for k, v in expected.items()):
        raise RuntimeError('original21 identity mismatch')
    import torch
    if torch.__version__ != '2.7.1+cu128' or torch.version.cuda != '12.8' or not torch.cuda.is_available():
        raise RuntimeError('tested Blackwell runtime required')
    gpu = torch.cuda.get_device_properties(0)
    if 'RTX PRO 6000' not in gpu.name.upper() or (gpu.major, gpu.minor) != (12, 0) or gpu.total_memory < 90 * 2**30:
        raise RuntimeError('requested PRO6000 Blackwell hardware required')
    checkpoint = assets / 'spynet_sintel_final-3d2a1287.pth'
    if sha(checkpoint) != '3d2a1287666aa71752ebaedc06999212886ef476f77d691a1b0006107088e714':
        raise RuntimeError('official SpyNet mismatch')
    record = dict(state='running', scope='two_discarded_native_frame_model_updates_ONLY',
                  started_at_unix=time.time(), job_id=os.environ['SLURM_JOB_ID'], GPU=gpu.name,
                  original21=expected, model_weights_retained=False, formal_AP=False)
    with output.open('x') as handle:
        json.dump(record, handle, indent=2)
    try:
        subprocess.run([sys.executable, str(original / 'probe_native.py'), '--checkpoint', checkpoint,
                        '--data-root', assets / 'kitti', '--records', assets / 'cao2025-roi-train-001.jsonl',
                        '--roi-audit', assets / 'cao2025-roi-train-audit-001.json', '--output', native], check=True)
        result = json.loads(native.read_text())
        if result['state'] != 'passed' or result['frame_id'] != '000000' or result['device'] != gpu.name:
            raise RuntimeError('actual pinned native model execution failed')
        if [(v['stage'], v['epoch'], v['active_parameter_tensors']) for v in result['phases']] != [(1, 1, 562), (5, 26, 718)]:
            raise RuntimeError('native phase/gradient scope mismatch')
        if any(not v['frozen_states_identical'] or not v['all_active_gradients_finite'] for v in result['phases']):
            raise RuntimeError('native gradients/read-only scope mismatch')
        if any(sha(original / k) != v for k, v in expected.items()):
            raise RuntimeError('original21 changed during job')
        record.update(state='passed_original_native_PRO6000_GPU_engineering',
                      native_result_sha256=sha(native), native_shape=result['native_shape'],
                      phase_results=result['phases'], torch=result['torch_version'],
                      sources21_before_after_identical=True,
                      checkpoint_sha256=result['checkpoint_sha256'], image_sha256=result['image_sha256'])
    except Exception as error:
        record.update(state='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        record['finished_at_unix'] = time.time()
        output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
