"""Full Table IX CNNs with sparse key transport and serialized received control."""
import torch
from torch import nn
from radio import (FrameErasure, decode_control, encode_control, equalize,
                   normalize_data, pilot_count, propagate, roi_masks, support_masks)


class ChannelCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            *[nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()) for _ in range(3)],
            nn.Conv2d(64, 9, 3, padding=1))
        self.decoder = nn.Sequential(nn.Conv2d(9, 64, 3, padding=1), nn.ReLU(),
            *[nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()) for _ in range(5)],
            nn.Conv2d(64, 3, 3, padding=1))


class WirelessVariant(nn.Module):
    def __init__(self, semantic):
        super().__init__()
        self.semantic = semantic
        # Sharing is fixed explicitly: global/key codecs independent, view-shared.
        self.global_channel = ChannelCodec()
        self.key_channel = ChannelCodec()

    def transmit_payload(self, payload, boxes, channel):
        shape = payload['original_shape']
        masks = roi_masks(shape, boxes, payload['global_values'][0])
        if any(not torch.equal(mask, saved) for mask, saved in zip(masks, payload['masks'])):
            raise ValueError('cached masks and transmitted boxes disagree')
        if len(payload['global_values']) != 2 or len(payload['key_values']) != 2:
            raise ValueError('two stereo semantic streams required')
        codes = [self.global_channel.encoder(value) for value in payload['global_values']]
        if any(code.shape[0] != 1 for code in codes):
            raise ValueError('native variable-size air frames require batch1')
        streams = [code.reshape(-1) for code in codes]
        supports = support_masks(masks)
        for value, support in zip(payload['key_values'], supports):
            code = self.key_channel.encoder(value)
            if code.shape != (1, 9, *support.shape[-2:]):
                raise ValueError('key code geometry mismatch')
            streams.append(code[0, :, support[0, 0]].reshape(-1))
        data = normalize_data(torch.cat(streams))
        control = encode_control(shape, boxes, data)
        npilots = pilot_count(channel)
        pilots = data.new_tensor([1., 0.]).expand(npilots, 2)
        air = torch.cat((pilots, control, data))
        accounting = dict(data_uses=len(data), control_uses=len(control), pilot_uses=npilots,
            total_uses=len(air), total_energy=air.detach().double().square().sum().item(),
            data_real_values=sum(stream.numel() for stream in streams),
            padding_real_values=sum(stream.numel() for stream in streams)%2,
            key_cells=[int(support.sum()) for support in supports],
            cbr_complex_per_rgb_real_value=len(air)/(2*3*shape[0]*shape[1]))
        return air, accounting

    def receive_payload(self, observation):
        # This signature intentionally accepts no clean masks, boxes, shape,
        # semantic tensors, normalization factor or actual fading coefficient.
        received = equalize(observation)
        shape, boxes, start = decode_control(received)
        h, w = shape
        hp, wp = h+(-h)%6, w+(-w)%6
        # Check count without allocating the received image masks first.
        global_count = 2*9*(hp//6)*(wp//6)
        if received.shape[0]-start < (global_count+1)//2:
            raise FrameErasure('too few received global data symbols')
        masks = roi_masks(shape, boxes, received)
        supports = support_masks(masks)
        sizes = [9*(hp//6)*(wp//6)]*2 + [9*int(support.sum()) for support in supports]
        nreal = sum(sizes)
        if received.shape[0]-start != (nreal+1)//2:
            raise FrameErasure('received duration disagrees with decoded layout')
        values = received[start:].reshape(-1)[:nreal]
        streams = values.split(sizes)
        globals_ = tuple(self.global_channel.decoder(value.reshape(1, 9, hp//6, wp//6))
                         for value in streams[:2])
        keys = []
        for value, support in zip(streams[2:], supports):
            # Scatter only received cells; no clean encoder values outside ROI.
            positions = support[0, 0].reshape(-1).nonzero().reshape(-1)
            canvas = value.new_zeros((9, (hp//2)*(wp//2)))
            canvas = canvas.scatter(1, positions.expand(9, -1), value.reshape(9, -1))
            keys.append(self.key_channel.decoder(canvas.reshape(1, 9, hp//2, wp//2)))
        return dict(global_values=globals_, key_values=tuple(keys), masks=masks, original_shape=shape)

    @staticmethod
    def semantic_mse(clean, received):
        # Channel-only stage targets frozen compressed semantic features. Key
        # reduction includes only transmitted ROI support, with empty ROI zero.
        terms = [(a-b).square().mean() for a, b in zip(clean['global_values'], received['global_values'])]
        for a, b, support in zip(clean['key_values'], received['key_values'], support_masks(clean['masks'])):
            if support.any():
                terms.append((a-b)[support.expand_as(a)].square().mean())
        return torch.stack(terms).mean()

    def forward(self, left, right, boxes, channel='awgn', snr_db=10., generator=None, decode_rgb=True):
        masks = roi_masks(left.shape[-2:], boxes, left)
        native_masks = tuple(mask[..., :left.shape[-2], :left.shape[-1]] for mask in masks)
        clean = self.semantic.encode(left, right, native_masks)
        air, accounting = self.transmit_payload(clean, boxes, channel)
        try:
            received = self.receive_payload(propagate(air, channel, snr_db, generator))
        except FrameErasure as error:
            # Caller records erasure; receiver never recovers with clean control.
            return dict(outputs=None, semantic_mse=None, accounting=accounting, erasure=str(error))
        return dict(outputs=self.semantic.decode(received) if decode_rgb else None,
            semantic_mse=self.semantic_mse(clean, received), accounting=accounting, erasure=None)
