"""Actual source spatial-rate variants; frozen30 model stays untouched."""
import math
import sys
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'reproduction/cao2025'))
# Import by unique file identity; this file itself is also named model.py.
import importlib.util
spec = importlib.util.spec_from_file_location('original_frozen_source_model',ROOT/'reproduction/cao2025/model.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
activated,conv,Residual = base.activated,base.conv,base.Residual
FACTORS = {10:(6,1),30:(6,2),50:(8,4)}


class GlobalEncoder(nn.Module):
    def __init__(self,factor):
        super().__init__()
        self.factor = factor
        self.stems = nn.ModuleList([nn.Sequential(activated(3,64),*[Residual() for _ in range(3)]) for _ in range(2)])
        self.joint = activated(128,64,1)
        self.branches = nn.ModuleList([nn.Sequential(activated(64,64,5),activated(64,4,5),
                                                   nn.PixelUnshuffle(factor),conv(4*factor*factor,3)) for _ in range(2)])

    def forward(self,left,right):
        common = self.joint(torch.cat((self.stems[0](left),self.stems[1](right)),1))
        return tuple(branch(common)+F.interpolate(image,scale_factor=1/self.factor,mode='bilinear',align_corners=False)
                     for branch,image in zip(self.branches,(left,right)))


class KeyEncoder(nn.Module):
    def __init__(self,factor):
        super().__init__()
        self.factor = factor
        self.layers = nn.Sequential(activated(3,6,5),nn.PixelUnshuffle(factor),conv(6*factor*factor,3))

    def forward(self,image):
        return self.layers(image)+F.interpolate(image,scale_factor=1/self.factor,mode='bilinear',align_corners=False)


class KeyDecoder(nn.Module):
    def __init__(self,factor):
        super().__init__()
        self.factor = factor
        self.layers = nn.Sequential(conv(3,6*factor*factor),nn.PixelShuffle(factor),nn.LeakyReLU(.1),activated(6,3))

    def forward(self,value):
        return self.layers(value)+F.interpolate(value,scale_factor=self.factor,mode='bilinear',align_corners=False)


class GlobalDecoder(base.GlobalDecoder):
    def __init__(self,flow,warp,factor):
        nn.Module.__init__(self)
        self.flow,self.warp,self.factor = flow,warp,factor
        self.initial = nn.ModuleList([nn.Sequential(activated(3,64),*[Residual() for _ in range(30)]) for _ in range(2)])
        self.recovery = nn.ModuleList([nn.Sequential(activated(67,64),*[Residual() for _ in range(30)]) for _ in range(2)])
        second = factor//2
        self.outputs = nn.ModuleList([nn.Sequential(activated(128,64,1),conv(64,256),nn.PixelShuffle(2),nn.LeakyReLU(.1),
            conv(64,64*second*second),nn.PixelShuffle(second),nn.LeakyReLU(.1),activated(64,64),conv(64,3)) for _ in range(2)])

    def forward(self,left,right):
        left_flow,right_flow = self.flow(left,right),self.flow(right,left)
        first_left,first_right = self.initial[0](left),self.initial[1](right)
        aligned_left = self.warp(first_right,left_flow.permute(0,2,3,1),padding_mode='border')
        aligned_right = self.warp(first_left,right_flow.permute(0,2,3,1),padding_mode='border')
        recovered_left = self.recovery[0](torch.cat((aligned_left,left),1))
        recovered_right = self.recovery[1](torch.cat((aligned_right,right),1))
        return tuple(output(torch.cat((initial,recovered),1))+F.interpolate(value,scale_factor=self.factor,mode='bilinear',align_corners=False)
                     for output,initial,recovered,value in zip(self.outputs,(first_left,first_right),(recovered_left,recovered_right),(left,right)))


class SemanticRateVariant(base.SemanticVariant):
    def __init__(self,spynet_checkpoint,nominal_rate):
        nn.Module.__init__(self)
        self.nominal_rate = nominal_rate
        self.global_factor,self.key_factor = FACTORS[nominal_rate]
        flow,warp = base.load_official_spynet(spynet_checkpoint)
        self.global_encoder = GlobalEncoder(self.global_factor)
        self.global_decoder = GlobalDecoder(flow,warp,self.global_factor)
        self.key_encoders = nn.ModuleList([KeyEncoder(self.key_factor),KeyEncoder(self.key_factor)])
        self.key_decoders = nn.ModuleList([KeyDecoder(self.key_factor),KeyDecoder(self.key_factor)])
        self.fusions = nn.ModuleList([nn.Sequential(activated(6,64,5),activated(64,64,5),conv(64,3)) for _ in range(2)])

    def padded_images(self,left,right):
        if left.shape != right.shape or left.ndim != 4 or left.shape[1] != 3:
            raise ValueError('equal Bx3xHxW stereo RGB required')
        h,w = left.shape[-2:]
        if min(h,w) <= 32*self.global_factor:
            raise ValueError('global images too small for official six-level SpyNet')
        factor = math.lcm(self.global_factor,self.key_factor)
        pad = (0,(-w)%factor,0,(-h)%factor)
        return F.pad(left,pad,mode='replicate'),F.pad(right,pad,mode='replicate'),(h,w)
