"""Trained six-state float32 SRCNN reception, byte geometry and float storage."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROTOCOL = HERE.parent / 'srcnn-float-reception-protocol-001.md'
PROTOCOL_SHA = '0626389a87f72979920dd08f8d7df173bf809ed87e5f3601ba448bb90fa5625d'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


interface = module('sealed_srcnn_float_source', HERE.parent / 'srcnn-compression-interface-code/interface.py')
training_model = module('sealed_srcnn_float_model', HERE.parent / 'srcnn-KITTI-adaptation-code-002/model.py')
SHAPES = {'weight1': (64, 1, 9, 9), 'bias1': (64,), 'weight2': (32, 64, 5, 5),
          'bias2': (32,), 'weight3': (1, 32, 5, 5), 'bias3': (1,)}


def sha_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


def describe(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous().numpy()
    value = np.ascontiguousarray(value)
    schema = dict(dtype=value.dtype.str, shape=list(value.shape))
    return dict(**schema, sha256=sha_bytes(json.dumps(schema, sort_keys=True).encode() + b'\0' + value.tobytes()))


def states(model):
    return {key: describe(value) for key, value in model.state_dict().items()}


def model_from_checkpoint(blob, run, expected_sources, scope, rate, device='cpu'):
    """Validate static checkpoint metadata; caller must enforce full closure gate.

    This lower-level function cannot authorize a formal run on its own. The
    native cache launcher must also require the terminal three-rate closure
    and complete local proof named in the protocol.
    """
    assert scope in ('engineering', 'formal') and rate in interface.RATES
    assert isinstance(blob, bytes) and sha_bytes(blob) == run['checkpoint_sha256']
    epochs, updates = (6, 6) if scope == 'engineering' else (20, 37120)
    assert run['state'] == 'finished' and run['scope'] == scope and run['rate'] == rate
    assert run['epochs'] == run['selected_epoch'] == epochs and run['optimizer_steps'] == updates
    assert run['sources'] == expected_sources
    checkpoint = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=True)
    assert checkpoint['rate'] == rate and checkpoint['epochs'] == epochs and checkpoint['optimizer_steps'] == updates
    assert checkpoint['sources'] == expected_sources and checkpoint['initial_state'] == run['initial_state']
    values = checkpoint['model']; assert set(values) == set(SHAPES)
    assert all(value.dtype == torch.float32 and tuple(value.shape) == SHAPES[key]
               and torch.isfinite(value).all() for key, value in values.items())
    assert {key: describe(value) for key, value in values.items()} == run['final_state']
    author = SimpleNamespace(kernel_sizes=(9, 5, 5), state_dict=lambda: values)
    model = training_model.Srcnn(author).to(device).eval().requires_grad_(False)
    assert len(model.state_dict()) == 6 and states(model) == run['final_state']
    assert all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters())
    return model


def receive(wire, model):
    assert not torch.is_grad_enabled() and all(not m.training for m in model.modules())
    assert all(not p.requires_grad for p in model.parameters())
    initial = states(model); assert set(initial) == set(SHAPES)
    low_views, header = interface.unpack(wire)
    height, width = header['native_hw']; outputs, records = [], []
    device = next(model.parameters()).device
    for low in low_views:
        colors = interface.rgb_to_ycbcr(interface.interpolate(low, height, width))
        value = torch.from_numpy(colors[:, :, 0].astype(np.float32)[None, None].copy()).to(device)
        predicted = model(value)
        assert predicted.dtype == torch.float32 and list(predicted.shape) == [1, 1, height, width]
        assert torch.isfinite(predicted).all()
        colors[:, :, 0] = predicted[0, 0].detach().cpu().numpy().astype(np.float64)
        rgb = np.ascontiguousarray(interface.ycbcr_to_rgb(colors), dtype='<f4')
        assert rgb.shape == (height, width, 3) and np.isfinite(rgb).all()
        outputs.append(rgb)
        records.append(dict(minimum=float(rgb.min()), maximum=float(rgb.max()),
                            out_of_range_fraction=float(np.mean((rgb < 0) | (rgb > 1)))))
    assert states(model) == initial
    return outputs, dict(**header, wire_sha256=sha_bytes(wire), readonly_states=initial,
                         arrays={name: describe(a) for name, a in zip(('left', 'right'), outputs)},
                         ranges=records, received_header_only_geometry=True, output_unclipped=True,
                         source_only_PHY_uses=None, source_only_PHY_energy=None)


def pack_pair(outputs):
    assert len(outputs) == 2 and outputs[0].shape == outputs[1].shape
    assert all(a.dtype == np.dtype('<f4') and a.ndim == 3 and a.shape[-1] == 3
               and np.isfinite(a).all() for a in outputs)
    buffer = io.BytesIO()
    np.savez(buffer, left=outputs[0], right=outputs[1])
    return buffer.getvalue()


def tensor_pair(blob, row):
    assert sha_bytes(blob) == row['cache_sha256']
    arrays = []
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        assert set(archive.files) == {'left', 'right'}
        for name in ('left', 'right'):
            value = archive[name]
            assert value.dtype == np.dtype('<f4') and value.ndim == 3 and value.shape[-1] == 3
            assert np.isfinite(value).all() and describe(value) == row['arrays'][name]
            assert list(value.shape[:2]) == row['native_hw']
            arrays.append(value.copy())
    assert arrays[0].shape == arrays[1].shape
    return tuple(torch.from_numpy(a.transpose(2, 0, 1).copy()).unsqueeze(0) for a in arrays)


def guard(allowed_inputs, control, calibration_inputs=()):
    """Read barrier after static model construction; reconstruction has no calib."""
    allowed = {str(Path(p).resolve()) for p in allowed_inputs}
    calibration = {str(Path(p).resolve()) for p in calibration_inputs}
    def barrier(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)) or not control['active']:
            return
        name = os.fsdecode(args[0]); mode, flags = args[1:3]
        reading = (isinstance(mode, str) and ('r' in mode or '+' in mode)) or (
            mode is None and flags & os.O_ACCMODE != os.O_WRONLY)
        if not reading:
            return
        assert not name.lower().endswith(('.png', '.pkl', '.pth', '.pt', '.mat', '.bin'))
        assert '/label_2/' not in name and '/velodyne/' not in name
        path = str(Path(name).resolve())
        if '/calib/' in name:
            assert path in calibration, 'foreign calibration read forbidden'
        if name.lower().endswith(('.npz', '.p6sr')):
            assert path in allowed, 'foreign received input forbidden'
    return barrier
