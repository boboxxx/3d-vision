"""Fixed ECSIC checkpoint recovery; integer fixtures are not compressed payloads."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import types

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path('/mnt/d/paper6/third_party/ecsic-696f4ae4')
WEIGHT = Path('/mnt/d/paper6/assets/ecsic-cs001-001/model.pt')
CONFIG = ROOT / 'data/provenance/ecsic-cs001-official-config-001.json'
MODEL_SHA = 'e66b55e2f888f5d113ea6c6414640c7486280d6e18838ba77c650720a5636052'
CONFIG_SHA = 'ba02cc1239b755a26ca0eb920c3014908c97c45fa210c447c47e96f7527102f8'
ORDER = ['z_left', 'z_right', 'y_left', 'y_right']


def check(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def arr(t):
    return t.detach().cpu().contiguous().numpy()


def describe(a):
    a = np.ascontiguousarray(a)
    return dict(shape=list(a.shape), dtype=str(a.dtype),
                sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                finite=bool(np.isfinite(a).all()), min=float(a.min()), max=float(a.max()))


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def state(model):
    return {k: describe(arr(v)) for k, v in model.state_dict().items()}


def load():
    check(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only environment required')
    check(os.environ.get('WANDB_MODE') == 'disabled', 'remote logger disabled')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(1907)
    check(sha(WEIGHT) == MODEL_SHA and sha(CONFIG) == CONFIG_SHA, 'public file identities')
    expected = json.loads((ROOT / 'data/provenance/ECSIC-official-source-audit-001.json').read_text())['files']
    check(all(sha(SOURCE / k) == v for k, v in expected.items()), 'official sources changed')
    package = types.ModuleType('ecsic')
    package.__path__ = [str(SOURCE / 'ecsic')]
    sys.modules['ecsic'] = package
    models = importlib.import_module('ecsic.models')
    utils = importlib.import_module('ecsic.utils')
    config = json.loads(CONFIG.read_text())['model']
    model = getattr(models, config['name'])(**config['kwargs']).cpu().eval()
    weights = torch.load(WEIGHT, map_location='cpu', weights_only=True)
    check(set(weights) == set(model.state_dict()), 'complete checkpoint key set')
    model.load_state_dict(weights, strict=True)
    check(all(torch.equal(weights[k], v) for k, v in model.state_dict().items()), 'loaded tensor identity')
    model.requires_grad_(False)
    versions = {n: importlib.import_module(n).__version__ for n in ('torch', 'numpy', 'torchvision', 'einops')}
    return model, utils, dict(model_sha256=MODEL_SHA, config_sha256=CONFIG_SHA,
                             official_sources=expected, versions=versions,
                             source_sha256=sha(Path(__file__)), state_count=len(weights),
                             torch_threads=torch.get_num_threads(), device='cpu')


def validate(streams, public):
    check(public['schema'] == 'ecsic-integer-fixture-v1', 'fixture version')
    check(public['stream_order'] == ORDER and list(streams) == ORDER, 'stream dependency order')
    check(public['model_sha256'] == MODEL_SHA and public['config_sha256'] == CONFIG_SHA, 'decoder identity')
    h, w = public['padded_hw']
    check(type(h) is int and type(w) is int and 0 < h <= 2048 and 0 < w <= 4096 and h % 32 == w % 32 == 0, 'public dimensions')
    for name, a in streams.items():
        divisor = 32 if name.startswith('z') else 8
        check(a.dtype == np.int32 and a.shape == (1, 48, h // divisor, w // divisor), 'integer stream layout: ' + name)
        check(np.max(np.abs(a.astype(np.int64))) < 2**24, 'FP32 exact integer range')


@torch.inference_mode()
def decode(model, utils, streams, public):
    validate(streams, public)
    h, w = public['padded_hw']
    pos = utils.get_positional_fourier_encoding(h, w).unsqueeze(0)
    q = {k: torch.from_numpy(v).float() for k, v in streams.items()}
    values = {}
    values['z_left_loc'], values['z_left_scale'] = model.zl_loc, model.zl_scale
    values['z_left_hat'] = q['z_left'] + model.zl_loc
    values['z_right_loc'], values['z_right_scale'] = model.zr_entropy(values['z_left_hat'])
    values['z_right_hat'] = q['z_right'] + values['z_right_loc']
    left, right, _, _ = model.HD(values['z_left_hat'], values['z_right_hat'], pos=pos, return_attn=True)
    values['y_left_loc'], values['y_left_scale'] = model.hd_out_left(left).chunk(2, 1)
    right = model.hd_out_right(right)
    values['y_left_hat'] = q['y_left'] + values['y_left_loc']
    loc, scale, _, _ = model.yr_entropy(values['y_left_hat'], right, pos=pos)
    values['y_right_loc'], values['y_right_scale'] = loc, scale
    values['y_right_hat'] = q['y_right'] + loc
    values['pred_left'], values['pred_right'], _, _ = model.D(values['y_left_hat'], values['y_right_hat'], pos=pos, return_attn=True)
    for name in ORDER:
        values[name + '_symbols'] = q[name].to(torch.int32)
    return {k: arr(v) for k, v in values.items()}


def hooks(model, reject_encoder=False):
    counts = dict(E=0, HE=0, HD=0, D=0)
    handles = []
    for name in counts:
        def callback(module, inputs, name=name):
            counts[name] += 1
            check(not (reject_encoder and name in ('E', 'HE')), 'receiver invoked source encoder')
        handles.append(getattr(model, name).register_forward_pre_hook(callback))
    return counts, handles


def receiver(directory):
    # Separate interpreter receives only the integer fixture and public header.
    def forbid_source_read(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            name = os.fsdecode(args[0])
            check('/data/kitti/' not in name and not name.endswith('reference.npz'), 'receiver attempted source/reference file access')
    sys.addaudithook(forbid_source_read)
    model, utils, report = load()
    initial = state(model)
    public = json.loads((directory / 'public.json').read_text())
    with np.load(directory / 'symbols.npz', allow_pickle=False) as data:
        streams = {k: data[k] for k in data.files}
    counts, _ = hooks(model, reject_encoder=True)
    outputs = decode(model, utils, streams, public)
    check(all(np.isfinite(a).all() for a in outputs.values()), 'nonfinite receiver output')
    check(state(model) == initial and all(p.grad is None for p in model.parameters()), 'receiver states/gradients changed')
    check(counts == dict(E=0, HE=0, HD=1, D=1), 'receiver call counts')
    np.savez(directory / 'decoded.npz', **outputs)
    report.update(state='passed', calls=counts, full_states_before=initial, full_states_after=state(model),
                  no_parameter_gradients=True, arrays={k: describe(v) for k, v in outputs.items()},
                  fixture_sha256=sha(directory / 'symbols.npz'), public_sha256=sha(directory / 'public.json'),
                  decoded_sha256=sha(directory / 'decoded.npz'), source_read_barrier=True)
    save_json(directory / 'receiver.json', report)


@torch.inference_mode()
def sender(model, utils, images, directory):
    shape = images[0].shape
    check(shape == images[1].shape and shape[2] == 3, 'paired RGB8 dimensions')
    h, w = shape[:2]
    ph, pw = (h + 31) // 32 * 32, (w + 31) // 32 * 32
    tensors = [F.pad(torch.from_numpy(a.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255,
                     (0, pw - w, 0, ph - h), mode='replicate') for a in images]
    pos = utils.get_positional_fourier_encoding(ph, pw).unsqueeze(0)
    ref = model(*tensors, pos=pos)
    yl, yr, _, _ = model.E(*tensors, pos=pos, return_attn=True)
    zl, zr, _, _ = model.HE(yl, yr, pos=pos, return_attn=True)
    originals = dict(z_left=zl, z_right=zr, y_left=yl, y_right=yr)
    symbols, reference = {}, {}
    for name in ORDER:
        level, side = name.split('_')
        lat = ref.latents[side]
        loc, scale = lat[level + '_loc'], lat[level + '_scale']
        q = torch.round(originals[name] - loc)
        check(torch.max(torch.abs(q)) < 2**24, 'integer range')
        check(torch.equal(q + loc, lat[level + '_hat_dec']), 'exact residual reconstruction: ' + name)
        check(torch.equal(torch.round(lat[level + '_hat_dec'] - loc), q), 'exact residual identity: ' + name)
        symbols[name] = arr(q.to(torch.int32))
        for suffix, tensor in [('loc', loc), ('scale', scale), ('hat', lat[level + '_hat_dec']), ('symbols', q.to(torch.int32))]:
            reference[name + '_' + suffix] = arr(tensor)
    reference.update(pred_left=arr(ref.pred.left), pred_right=arr(ref.pred.right))
    public = dict(schema='ecsic-integer-fixture-v1', stream_order=ORDER, padded_hw=[ph, pw],
                  original_hw=[h, w], padding_lrtb=[0, pw-w, 0, ph-h],
                  model_sha256=MODEL_SHA, config_sha256=CONFIG_SHA)
    validate(symbols, public)
    failures = []
    bad_cases = [({k: v for k, v in symbols.items() if k != 'y_right'}, public),
                 (dict(reversed(list(symbols.items()))), public),
                 ({**symbols, 'z_left': symbols['z_left'][:, :, :, :-1]}, public),
                 ({**symbols, 'z_left': symbols['z_left'].astype(np.float32)}, public)]
    for i, (bad, header) in enumerate(bad_cases):
        try:
            validate(bad, header)
        except RuntimeError:
            failures.append(i)
    check(len(failures) == 4, 'malformed input not rejected')
    np.savez(directory / 'symbols.npz', **symbols)
    np.savez(directory / 'reference.npz', **reference)
    save_json(directory / 'public.json', public)
    return dict(input_arrays=[describe(a) for a in images], public=public,
                malformed_cases_rejected=failures, arrays={k: describe(a) for k, a in reference.items()})


def run(directory):
    from PIL import Image
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(state='running', started_unix=time.time(), pid=os.getpid(), cases={})
    save_json(directory / 'manifest.json', report)
    try:
        model, utils, meta = load()
        initial = state(model)
        report.update(meta)
        counts, _ = hooks(model)
        rng = np.random.Generator(np.random.PCG64(1907))
        synthetic = [rng.integers(0, 256, size=(32, 64, 3), dtype=np.uint8) for _ in range(2)]
        for case in ('synthetic32x64', 'training000000'):
            if case == 'synthetic32x64':
                images, inputs = synthetic, []
            else:
                inputs = [Path('/mnt/d/paper6/data/kitti/training') / side / '000000.png' for side in ('image_2', 'image_3')]
                images = []
                for path in inputs:
                    with Image.open(path) as image:
                        images.append(np.array(image.convert('RGB')))
            target = directory / case
            target.mkdir()
            print(json.dumps(dict(case=case, stage='sender')), flush=True)
            evidence = sender(model, utils, images, target)
            evidence['input_files'] = {str(p): sha(p) for p in inputs}
            save_json(target / 'sender.json', evidence)
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--receiver', '--directory', str(target)],
                                    capture_output=True, text=True, env=os.environ.copy())
            (target / 'receiver.log').write_text(result.stdout + result.stderr)
            check(result.returncode == 0, 'receiver subprocess failed: ' + case)
            with np.load(target / 'reference.npz', allow_pickle=False) as ref, np.load(target / 'decoded.npz', allow_pickle=False) as dec:
                check(set(ref.files) == set(dec.files), 'full receiver array scope')
                equal = {k: bool(np.array_equal(ref[k], dec[k])) for k in ref.files}
                check(all(equal.values()), 'receiver exact equality failed: ' + str(equal))
            receiver_report = json.loads((target / 'receiver.json').read_text())
            check(receiver_report['state'] == 'passed', 'receiver report failed')
            report['cases'][case] = dict(state='passed', exact_arrays=equal, public=evidence['public'],
                                         malformed_cases_rejected=evidence['malformed_cases_rejected'],
                                         files={p.name: sha(p) for p in target.iterdir()})
            save_json(directory / 'manifest.json', report)
            print(json.dumps(dict(case=case, state='passed', exact_arrays=len(equal))), flush=True)
        check(state(model) == initial and all(p.grad is None for p in model.parameters()), 'sender states/gradients changed')
        check(counts == dict(E=4, HE=4, HD=2, D=2), 'sender calls')
        check(sha(WEIGHT) == MODEL_SHA and sha(CONFIG) == CONFIG_SHA, 'public assets changed')
        check(all(sha(SOURCE / k) == v for k, v in meta['official_sources'].items()), 'official sources changed')
        report.update(state='passed', finished_unix=time.time(), calls=counts,
                      full_states_before=initial, full_states_after=state(model), no_parameter_gradients=True,
                      protocol_sha256=sha(ROOT / 'experiments/original-paper-scenarios/ecsic-recovery-protocol-001.md'),
                      actual_entropy_coding=False, compression_or_AP_claim=False)
    except Exception:
        report.update(state='failed', failed_unix=time.time(), traceback=traceback.format_exc())
        raise
    finally:
        save_json(directory / 'manifest.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--receiver', action='store_true')
    args = parser.parse_args()
    (receiver if args.receiver else run)(args.directory)
