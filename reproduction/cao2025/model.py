"""Complete semantic/flow architecture for the explicitly documented variant.

This is separate from the new direct-detection method. Returned values are RGB;
trained YOLO ROIs and pretrained SpyNet must come from external sensor modules.
"""
import ast
import math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F


def load_official_spynet(checkpoint):
    root = Path(__file__).parent/'upstream'
    spynet = ast.parse((root/'spynet_arch.py').read_text())
    util = ast.parse((root/'arch_util.py').read_text())
    nodes = [node for node in util.body if isinstance(node, ast.FunctionDef) and node.name == 'flow_warp']
    for node in spynet.body:
        if isinstance(node, ast.ClassDef) and node.name in ('BasicModule', 'SpyNet'):
            node.decorator_list = []
            nodes.append(node)
    if len(nodes) != 3:
        raise RuntimeError('official upstream definitions differ')
    namespace = dict(math=math, torch=torch, nn=nn, F=F)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(root/'spynet_arch.py'), 'exec'), namespace)
    flow = namespace['SpyNet']()
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if 'params' not in weights:
        raise ValueError('official SpyNet params checkpoint required')
    state = dict(weights['params'])
    # The author release stores all 60 learned tensors, omitting fixed RGB
    # normalization buffers. Upstream loads before registering these buffers.
    # Restore only those two known constants and still require a strict match.
    for name in ('mean', 'std'):
        if name not in state:
            state[name] = flow.state_dict()[name].clone()
    flow.load_state_dict(state, strict=True)
    return flow, namespace['flow_warp']


def conv(cin, cout, kernel=3):
    return nn.Conv2d(cin, cout, kernel, padding=kernel//2)


def activated(cin, cout, kernel=3):
    return nn.Sequential(conv(cin, cout, kernel), nn.LeakyReLU(.1))


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(conv(64, 64), nn.ReLU(), conv(64, 64))

    def forward(self, x):
        return x + self.layers(x)


class GlobalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stems = nn.ModuleList([nn.Sequential(activated(3, 64), *[Residual() for _ in range(3)])
                                   for _ in range(2)])
        self.joint = activated(128, 64, 1)
        self.branches = nn.ModuleList([nn.Sequential(activated(64, 64, 5), activated(64, 4, 5),
            nn.PixelUnshuffle(6), conv(144, 3)) for _ in range(2)])

    def forward(self, left, right):
        common = self.joint(torch.cat((self.stems[0](left), self.stems[1](right)), 1))
        return tuple(branch(common) + F.interpolate(image, scale_factor=1/6, mode='bilinear', align_corners=False)
                     for branch, image in zip(self.branches, (left, right)))


class KeyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(activated(3, 6, 5), nn.PixelUnshuffle(2), conv(24, 3))

    def forward(self, image):
        return self.layers(image) + F.interpolate(image, scale_factor=.5, mode='bilinear', align_corners=False)


class KeyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(conv(3, 24), nn.PixelShuffle(2), nn.LeakyReLU(.1), activated(6, 3))

    def forward(self, value):
        return self.layers(value) + F.interpolate(value, scale_factor=2, mode='bilinear', align_corners=False)


class GlobalDecoder(nn.Module):
    def __init__(self, flow, warp):
        super().__init__()
        self.flow = flow
        self.warp = warp
        self.initial = nn.ModuleList([nn.Sequential(activated(3, 64), *[Residual() for _ in range(30)])
                                      for _ in range(2)])
        self.recovery = nn.ModuleList([nn.Sequential(activated(67, 64), *[Residual() for _ in range(30)])
                                       for _ in range(2)])
        self.outputs = nn.ModuleList([nn.Sequential(activated(128, 64, 1), conv(64, 256),
            nn.PixelShuffle(2), nn.LeakyReLU(.1), conv(64, 576), nn.PixelShuffle(3),
            nn.LeakyReLU(.1), activated(64, 64), conv(64, 3)) for _ in range(2)])

    def forward(self, left, right):
        # Official SpyNet(ref, supp) returns the sampling displacement in supp
        # coordinates for the reference view; do not swap forward/backward flow.
        left_flow, right_flow = self.flow(left, right), self.flow(right, left)
        first_left, first_right = self.initial[0](left), self.initial[1](right)
        aligned_left = self.warp(first_right, left_flow.permute(0, 2, 3, 1), padding_mode='border')
        aligned_right = self.warp(first_left, right_flow.permute(0, 2, 3, 1), padding_mode='border')
        recovered_left = self.recovery[0](torch.cat((aligned_left, left), 1))
        recovered_right = self.recovery[1](torch.cat((aligned_right, right), 1))
        return tuple(output(torch.cat((initial, recovered), 1))
                     + F.interpolate(value, scale_factor=6, mode='bilinear', align_corners=False)
                     for output, initial, recovered, value in zip(self.outputs,
                         (first_left, first_right), (recovered_left, recovered_right), (left, right)))


class SemanticVariant(nn.Module):
    def __init__(self, spynet_checkpoint):
        super().__init__()
        flow, warp = load_official_spynet(spynet_checkpoint)
        self.global_encoder = GlobalEncoder()
        self.global_decoder = GlobalDecoder(flow, warp)
        self.key_encoders = nn.ModuleList([KeyEncoder(), KeyEncoder()])
        self.key_decoders = nn.ModuleList([KeyDecoder(), KeyDecoder()])
        self.fusions = nn.ModuleList([nn.Sequential(activated(6, 64, 5), activated(64, 64, 5), conv(64, 3))
                                     for _ in range(2)])

    @staticmethod
    def padded_images(left, right):
        if left.shape != right.shape or left.ndim != 4 or left.shape[1] != 3:
            raise ValueError('equal Bx3xHxW stereo RGB required')
        h, w = left.shape[-2:]
        if min(h, w) <= 192:
            raise ValueError('global images too small for official six-level SpyNet')
        pad = (0, (-w)%6, 0, (-h)%6)
        return F.pad(left, pad, mode='replicate'), F.pad(right, pad, mode='replicate'), (h, w)

    def encode(self, left, right, masks):
        left, right, shape = self.padded_images(left, right)
        if len(masks) != 2 or any(mask.shape != (left.shape[0], 1, *shape) for mask in masks):
            raise ValueError('explicit sensor-generated per-view ROI masks required')
        h, w = shape
        masks = tuple(F.pad(mask, (0, left.shape[-1]-w, 0, left.shape[-2]-h)) for mask in masks)
        global_values = self.global_encoder(left, right)
        key_values = tuple(encoder(image*mask) for encoder, image, mask
                           in zip(self.key_encoders, (left, right), masks))
        return dict(global_values=global_values, key_values=key_values, masks=masks, original_shape=shape)

    def decode(self, payload):
        # Receiver receives compressed values and explicit ROI control, never
        # original RGB or clean sender features. Channel packing is a next gate.
        globals_ = self.global_decoder(*payload['global_values'])
        keys = tuple(decoder(value)*mask for decoder, value, mask
                     in zip(self.key_decoders, payload['key_values'], payload['masks']))
        h, w = payload['original_shape']
        return tuple((fusion(torch.cat((key, global_), 1)) + global_)[..., :h, :w]
                     for fusion, key, global_ in zip(self.fusions, keys, globals_))

    def forward(self, left, right, masks):
        return self.decode(self.encode(left, right, masks))

    def global_warmup(self, left, right):
        left, right, (h, w) = self.padded_images(left, right)
        globals_ = self.global_decoder(*self.global_encoder(left, right))
        return tuple((fusion(torch.cat((torch.zeros_like(value), value), 1))+value)[..., :h, :w]
                     for fusion, value in zip(self.fusions, globals_))
