"""CPU structural/physical checks, explicitly not trained AP or channel quality."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from model import SemanticVariant
from radio import (FrameErasure, Observation, PREFIX_USES, REPETITION,
                   decode_control, encode_control, equalize, normalize_data, propagate)
from wireless import WirelessVariant


def expect_erasure(fn):
    try:
        fn()
    except FrameErasure:
        return
    raise AssertionError('invalid received frame silently accepted')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(17)
    shape = (197, 203)
    boxes = ([dict(xyxy=[40, 30, 160, 150], confidence=.8)],
             [dict(xyxy=[35, 35, 155, 155], confidence=.75)])
    control = encode_control(shape, boxes, torch.zeros(1))
    assert len(control) == (13+4+2*12+4)*8*7
    decoded_shape, decoded_boxes, offset = decode_control(control)
    assert decoded_shape == shape and offset == len(control)
    for original, received in zip(boxes, decoded_boxes):
        assert original[0]['xyxy'] == received[0]['xyxy']
        assert abs(original[0]['confidence']-received[0]['confidence']) < 1e-7
    # Seven flipped copies corrupt a bit and must fail CRC, even with valid
    # sender control still in scope. One flipped copy is corrected by majority.
    damaged = control.clone()
    damaged[:REPETITION, 0] *= -1
    expect_erasure(lambda: decode_control(damaged))
    damaged = control.clone()
    damaged[PREFIX_USES:PREFIX_USES+REPETITION, 0] *= -1
    expect_erasure(lambda: decode_control(damaged))
    corrected = control.clone()
    corrected[0, 0] *= -1
    assert decode_control(corrected)[0] == shape
    odd = normalize_data(torch.tensor([1., 2., 3.]))
    assert odd.shape == (2, 2) and odd[-1, 1] == 0
    assert torch.allclose(odd.square().sum(), torch.tensor(2.))
    # Independent analytic fading fixture: estimate derived from pilots alone.
    fading = torch.tensor([.4, -.7])
    sent = torch.cat((torch.tensor([1., 0.]).expand(8, 2), torch.randn(10, 2)))
    r, i = sent.unbind(-1)
    faded = torch.stack((r*fading[0]-i*fading[1], r*fading[1]+i*fading[0]), -1)
    assert torch.allclose(equalize(Observation(faded, 0., 'rayleigh')), sent[8:], atol=1e-6)
    expect_erasure(lambda: equalize(Observation(torch.zeros(9, 2), 0., 'rayleigh')))
    noise = propagate(torch.zeros(100000, 2), 'awgn', 10.).symbols
    assert abs(noise.var().item()-.05) < .001
    model = WirelessVariant(SemanticVariant(args.checkpoint))
    left, right = torch.rand(1, 3, *shape), torch.rand(1, 3, *shape)
    outcome = model(left, right, boxes, channel='awgn', snr_db=10.)
    assert outcome['erasure'] is None
    assert all(value.shape == left.shape and torch.isfinite(value).all() for value in outcome['outputs'])
    account = outcome['accounting']
    # The odd-coordinate second box touches61x61 pooled cells, not60x60.
    # Boundary support must be charged even when the box area is unchanged.
    assert account['key_cells'] == [3600, 3721]
    assert account['data_real_values'] == 2*9*33*34+9*(3600+3721)
    assert account['total_uses'] == account['data_uses']+account['control_uses']+account['pilot_uses']
    assert abs(account['total_energy']-account['total_uses'])/account['total_uses'] < 1e-6
    loss = outcome['semantic_mse'] + sum((a-b).square().mean() for a, b in zip(outcome['outputs'], (left, right)))
    loss.backward()
    groups = {}
    for name, value in model.named_parameters():
        assert value.grad is not None and torch.isfinite(value.grad).all(), name
        group = '.'.join(name.split('.')[:2]) if name.startswith('semantic.') else name.split('.')[0]
        groups.setdefault(group, dict(parameters=0, gradient_squared_norm=0.))
        groups[group]['parameters'] += value.numel()
        groups[group]['gradient_squared_norm'] += value.grad.double().square().sum().item()
    assert all(group['gradient_squared_norm'] > 0 for group in groups.values())
    model.eval()
    with torch.no_grad():
        from radio import roi_masks
        masks = tuple(mask[..., :shape[0], :shape[1]] for mask in roi_masks(shape, ([], []), left))
        clean = model.semantic.encode(left, right, masks)
        air, empty_account = model.transmit_payload(clean, ([], []), 'rayleigh')
        assert empty_account['key_cells'] == [0, 0]
        assert empty_account['control_uses'] == (13+4+4)*8*7
        assert empty_account['pilot_uses'] == 8
        assert empty_account['data_real_values'] == 2*9*33*34
        assert abs(empty_account['total_energy']-len(air))/len(air) < 1e-6
        # Fixed channel fixture uses only actual received pilots at the decoder.
        r, i = air.unbind(-1)
        faded = torch.stack((r*fading[0]-i*fading[1], r*fading[1]+i*fading[0]), -1)
        received = model.receive_payload(Observation(faded, 0., 'rayleigh'))
        assert all(mask.sum() == 0 for mask in received['masks'])
        assert received['original_shape'] == shape
        broken = faded.clone()
        broken[8:8+REPETITION] *= -1
        expect_erasure(lambda: model.receive_payload(Observation(broken, 0., 'rayleigh')))
        expect_erasure(lambda: model.receive_payload(Observation(faded[:-1], 0., 'rayleigh')))
    root = Path(__file__).parent
    record = dict(status='passed', scope='untrained CPU wireless engineering; no RGB quality or detection AP',
        torch_version=torch.__version__, seed=17, shape=shape, parameters=sum(p.numel() for p in model.parameters()),
        parameter_tensors=sum(1 for _ in model.parameters()), groups=groups,
        awgn_real_noise_variance=noise.var().item(), control_crc_corruption_rejected=True,
        single_repeat_error_corrected=True, no_clean_control_receiver_api=True,
        missing_symbol_rejected=True, pilot_only_fading_fixture_passed=True,
        odd_real_padding_counted=True, all_parameter_gradients_finite=True,
        synthetic_untrained_loss=loss.item(), roi_accounting=account, empty_roi_accounting=empty_account,
        source_hashes={name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                       for name in ('model.py', 'radio.py', 'wireless.py', 'probe_wireless.py')},
        spynet_checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        elapsed_seconds=time.time()-started)
    Path(args.output).write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
