"""Engineering gates only; this does not measure trained communication AP."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
import torch
from model import SemanticVariant, load_official_spynet


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(17)
    started = time.time()
    root = Path(__file__).parent
    sources = json.loads((root/'upstream/sources.json').read_text())
    for name, expected in sources['files'].items():
        assert digest(root/'upstream'/name) == expected, name
    released = torch.load(args.checkpoint, map_location='cpu', weights_only=True)['params']
    flow, warp = load_official_spynet(args.checkpoint)
    learned = dict(flow.named_parameters())
    assert set(released) == set(learned), 'exact learned tensor coverage required'
    assert all(torch.equal(value, learned[name]) for name, value in released.items())
    assert len(released) == 60
    # Sampling convention: +1 horizontal displacement samples the right pixel.
    ramp = torch.arange(5.).view(1, 1, 1, 5).expand(1, 1, 3, 5).clone()
    displacement = torch.zeros(1, 3, 5, 2)
    displacement[..., 0] = 1
    expected = torch.tensor([1., 2., 3., 4., 4.]).view(1, 1, 1, 5).expand_as(ramp)
    assert torch.allclose(warp(ramp, displacement, padding_mode='border'), expected)
    # A missing learned tensor must fail, rather than retaining random weights.
    with tempfile.TemporaryDirectory() as tmp:
        broken = dict(released)
        del broken[next(iter(broken))]
        path = Path(tmp)/'missing.pth'
        torch.save({'params': broken}, path)
        try:
            load_official_spynet(path)
        except RuntimeError:
            pass
        else:
            raise AssertionError('missing learned flow tensor accepted')
    model = SemanticVariant(args.checkpoint)
    left, right = torch.rand(1, 3, 197, 203), torch.rand(1, 3, 197, 203)
    masks = [torch.zeros(1, 1, 197, 203) for _ in range(2)]
    masks[0][..., 30:150, 40:160] = 1
    masks[1][..., 35:155, 35:155] = 1
    payload = model.encode(left, right, masks)
    assert set(payload) == {'global_values', 'key_values', 'masks', 'original_shape'}
    assert all(value.shape == (1, 3, 33, 34) for value in payload['global_values'])
    assert all(value.shape == (1, 3, 99, 102) for value in payload['key_values'])
    outputs = model.decode(payload)
    assert all(value.shape == left.shape and torch.isfinite(value).all() for value in outputs)
    loss = sum((value-target).square().mean() for value, target in zip(outputs, (left, right)))
    loss.backward()
    parameters = dict(model.named_parameters())
    assert all(value.grad is not None and torch.isfinite(value.grad).all()
               for value in parameters.values()), 'missing/nonfinite full network gradient'
    groups = {}
    for name, value in parameters.items():
        group = name.split('.')[0]
        groups.setdefault(group, {'parameters': 0, 'gradient_squared_norm': 0.})
        groups[group]['parameters'] += value.numel()
        groups[group]['gradient_squared_norm'] += value.grad.double().square().sum().item()
    assert all(value['gradient_squared_norm'] > 0 for value in groups.values())
    with torch.no_grad():
        warmup = model.global_warmup(left, right)
    assert all(value.shape == left.shape and torch.isfinite(value).all() for value in warmup)
    record = dict(status='passed', scope='synthetic semantic variant engineering only; no channel or AP',
        device='cpu', torch_version=torch.__version__, seed=17,
        checkpoint_sha256=digest(args.checkpoint), checkpoint_bytes=Path(args.checkpoint).stat().st_size,
        upstream_sources_verified=True, learned_flow_tensors_exactly_loaded=60,
        fixed_flow_buffers=['mean', 'std'], missing_learned_weight_rejected=True,
        warp_positive_x_samples_right=True, original_shape=list(left.shape),
        padded_shape=[1, 3, 198, 204], output_shapes=[list(value.shape) for value in outputs],
        parameter_tensors=len(parameters), parameters=sum(value.numel() for value in parameters.values()),
        all_parameter_gradients_finite=True, parameter_groups=groups,
        global_warmup_finite=True, synthetic_untrained_loss=loss.item(),
        source_hashes={name: digest(root/name) for name in ('model.py', 'probe.py', 'upstream/sources.json')},
        elapsed_seconds=time.time()-started)
    Path(args.output).write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
