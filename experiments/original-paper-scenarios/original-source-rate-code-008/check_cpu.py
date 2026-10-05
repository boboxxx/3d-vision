"""Prelocked source-stage loss/gradient check; no training or wireless claim."""
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from source_loss import SemanticRateVariant, architecture, forward_loss, masks
from wireless import WirelessVariant
from stages import configure, phase, forward_loss as frozen_loss

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CONDITIONS = [(1, 1), (1, 3), (2, 1), (3, 1), (4, 1), (4, 6)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def states(model):
    return {k: digest(v) for k, v in model.state_dict().items()}


def check(model, left, right, boxes, loss_function):
    before = states(model)
    results = []
    for stage, epoch in CONDITIONS:
        current = phase(stage, epoch)
        active = configure(model, current)
        model.zero_grad(set_to_none=True)
        result = loss_function(model, left, right, boxes, current)
        output = result['outputs']
        assert all(v.shape == left.shape and bool(torch.isfinite(v).all()) for v in output)
        loss = result['loss']
        assert loss.ndim == 0 and bool(torch.isfinite(loss)) and float(loss.detach()) >= 0
        loss.backward()
        gradients = {k: p.grad for k, p in model.named_parameters() if p.grad is not None}
        assert gradients.keys() == active and active
        assert all(bool(torch.isfinite(v).all()) for v in gradients.values())
        assert states(model) == before
        results.append({'stage': stage, 'epoch': epoch, 'loss_type': current['loss'],
            'loss': float(loss.detach()), 'loss_sha256': digest(loss),
            'output_sha256': [digest(v) for v in output],
            'active_gradient_tensors': len(active),
            'gradient_values': sum(v.numel() for v in gradients.values()),
            'whole_gradient_ledger_sha256': hashlib.sha256(json.dumps(
                {k: digest(v) for k, v in gradients.items()}, sort_keys=True).encode()).hexdigest(),
            'states_unchanged': True})
        del result, output, loss, gradients
        print(json.dumps({'stage': stage, 'epoch': epoch, 'checked': True}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    result = {'state': 'starting', 'started_unix': time.time(),
        'scope': 'synthetic263x265_source_losses_and_complete_gradients_no_updates',
        'training_AP_radio_or_cross_host_bit_identity_claimed': False}
    try:
        lock = json.loads((HERE/'inputs.json').read_text())
        assert all(sha(ROOT/p) == expected for p, expected in lock['files'].items())
        torch.set_num_threads(2)
        rng = np.random.default_rng(17)
        left, right = [torch.from_numpy(rng.random((1, 3, 263, 265), dtype=np.float32)) for _ in range(2)]
        boxes = [[{'xyxy': [20, 30, 181, 193], 'confidence': .8}],
                 [{'xyxy': [17, 33, 177, 191], 'confidence': .75}]]
        result.update(raw_input_sha256=[digest(v) for v in [left, right]], boxes=boxes,
            inputs_sha256=sha(HERE/'inputs.json'), torch_version=torch.__version__, numpy_version=np.__version__)
        asset = ROOT/lock['spynet_path']
        assert sha(asset) == lock['spynet_sha256']
        torch.manual_seed(17)
        control = WirelessVariant(architecture.base.SemanticVariant(asset))
        control_states = states(control)
        expected = check(control, left, right, boxes, frozen_loss)
        del control
        result['rates'] = []
        for rate in [30, 10, 50]:
            torch.manual_seed(17)
            model = WirelessVariant(SemanticRateVariant(asset, rate))
            if rate == 30:
                assert states(model) == control_states
            actual = check(model, left, right, boxes, forward_loss)
            if rate == 30:
                assert actual == expected, 'Frozen30 output/loss/full gradients must be bit-identical'
            native, padded = masks((263, 265), boxes, left,
                model.semantic.global_factor, model.semantic.key_factor)
            factor = 8 if rate == 50 else 6
            hp, wp = 263+(-263)%factor, 265+(-265)%factor
            assert all(v.shape == (1, 1, hp, wp) for v in padded)
            assert all(torch.equal(a, b[..., :263, :265]) for a, b in zip(native, padded))
            assert all(not bool(v[..., 263:, :].any()) and not bool(v[..., :, 265:].any()) for v in padded)
            result['rates'].append({'nominal_rate': rate, 'padded_hw': [hp, wp],
                'complete_source_loss_gradient_conditions': actual,
                'nominal30_bit_identical_frozen_control': rate == 30})
            del model, actual, native, padded
        assert all(sha(ROOT/p) == expected for p, expected in lock['files'].items())
        result['state'] = 'passed_all3_six_source_phases_complete_gradients_and_frozen30_bit_identity'
    except BaseException:
        result.update(state='failed_retained', traceback=traceback.format_exc())
        raise
    finally:
        result['ended_unix'] = time.time()
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
