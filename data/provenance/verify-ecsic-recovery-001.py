"""Independent complete saved-array/checkpoint audit; no model forward or fitting."""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
RUN = Path('/mnt/d/paper6/runs/ecsic-recovery-stageA-001')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_text())


def describe(a):
    return dict(shape=list(a.shape), dtype=str(a.dtype), sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest(),
                finite=bool(np.isfinite(a).all()), min=float(a.min()), max=float(a.max()))


def main():
    output = ROOT / 'data/provenance/ecsic-recovery-stageA-001-audit.json'
    assert not output.exists()
    run = read(RUN / 'manifest.json')
    assert run['state'] == 'passed' and run['state_count'] == 225
    try:
        os.kill(run['pid'], 0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError('native process still live')
    assert run['source_sha256'] == sha(ROOT / 'experiments/original-paper-scenarios/ecsic-recovery-code/recover.py')
    assert run['protocol_sha256'] == sha(ROOT / 'experiments/original-paper-scenarios/ecsic-recovery-protocol-001.md')
    weight = Path('/mnt/d/paper6/assets/ecsic-cs001-001/model.pt')
    assert sha(weight) == run['model_sha256'] == 'e66b55e2f888f5d113ea6c6414640c7486280d6e18838ba77c650720a5636052'
    tensors = torch.load(weight, map_location='cpu', weights_only=True)
    actual = {k: describe(v.detach().contiguous().numpy()) for k, v in tensors.items()}
    assert actual == run['full_states_before'] == run['full_states_after'] and len(actual) == 225
    assert run['calls'] == dict(E=4, HE=4, HD=2, D=2) and run['no_parameter_gradients']
    assert not run['actual_entropy_coding'] and not run['compression_or_AP_claim']
    assert run['device'] == 'cpu' and run['torch_threads'] == 2
    source = Path('/mnt/d/paper6/third_party/ecsic-696f4ae4')
    assert all(sha(source / k) == v for k, v in run['official_sources'].items())
    order = ['z_left', 'z_right', 'y_left', 'y_right']
    rng = np.random.Generator(np.random.PCG64(1907))
    synthetic = [rng.integers(0, 256, (32, 64, 3), dtype=np.uint8) for _ in range(2)]
    results = {}
    for case in ('synthetic32x64', 'training000000'):
        d = RUN / case
        item = run['cases'][case]
        assert item['state'] == 'passed' and item['malformed_cases_rejected'] == [0, 1, 2, 3]
        assert all(sha(d / k) == v for k, v in item['files'].items())
        sender, receiver, public = read(d / 'sender.json'), read(d / 'receiver.json'), read(d / 'public.json')
        assert receiver['state'] == 'passed' and receiver['source_read_barrier']
        assert receiver['calls'] == dict(E=0, HE=0, HD=1, D=1)
        assert receiver['full_states_before'] == receiver['full_states_after'] == actual
        assert receiver['no_parameter_gradients'] and receiver['source_sha256'] == run['source_sha256']
        assert public == item['public'] == sender['public'] and public['stream_order'] == order
        assert receiver['fixture_sha256'] == sha(d / 'symbols.npz')
        assert receiver['public_sha256'] == sha(d / 'public.json') and receiver['decoded_sha256'] == sha(d / 'decoded.npz')
        if case == 'synthetic32x64':
            inputs = synthetic
        else:
            inputs = []
            paths = [Path('/mnt/d/paper6/data/kitti/training') / s / '000000.png' for s in ('image_2', 'image_3')]
            assert sender['input_files'] == {str(p): sha(p) for p in paths}
            for p in paths:
                with Image.open(p) as im:
                    inputs.append(np.array(im.convert('RGB')))
        assert [describe(a) for a in inputs] == sender['input_arrays']
        assert public['original_hw'] == list(inputs[0].shape[:2])
        h, w = public['padded_hw']
        with np.load(d / 'symbols.npz', allow_pickle=False) as sy, np.load(d / 'reference.npz', allow_pickle=False) as ref, np.load(d / 'decoded.npz', allow_pickle=False) as dec:
            assert sy.files == order
            expected = {name + '_' + suffix for name in order for suffix in ('loc', 'scale', 'hat', 'symbols')} | {'pred_left', 'pred_right'}
            assert set(ref.files) == set(dec.files) == set(item['exact_arrays']) == expected
            assert {k: describe(ref[k]) for k in ref.files} == sender['arrays']
            assert {k: describe(dec[k]) for k in dec.files} == receiver['arrays']
            for key in expected:
                assert np.array_equal(ref[key], dec[key]) and np.isfinite(dec[key]).all() and item['exact_arrays'][key]
            for name in order:
                factor = 32 if name.startswith('z') else 8
                assert sy[name].dtype == np.int32 and sy[name].shape == (1, 48, h // factor, w // factor)
                assert np.array_equal(sy[name], dec[name + '_symbols'])
                assert np.array_equal(sy[name].astype(np.float32) + dec[name + '_loc'], dec[name + '_hat'])
                assert np.array_equal(np.rint(dec[name + '_hat'] - dec[name + '_loc']).astype(np.int32), sy[name])
            assert dec['pred_left'].shape == dec['pred_right'].shape == (1, 3, h, w)
        results[case] = dict(exact_arrays=len(expected), files=item['files'], padded_hw=[h, w])
    result = dict(state='passed', actual_native_terminal=True, native_pid=run['pid'],
                  manifest_sha256=sha(RUN / 'manifest.json'), verifier_sha256=sha(Path(__file__)),
                  original_state_tensors=225, cases=results, audited_unix=time.time(),
                  no_new_model_forward=True, actual_entropy_coding=False)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'state': 'passed', 'cases': 2, 'arrays_per_case': 18, 'state_tensors': 225}))


if __name__ == '__main__':
    main()
