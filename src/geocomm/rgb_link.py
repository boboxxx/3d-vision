"""Matched analog RGB reconstruction comparator using the same channel.

New controlled comparison; not Cao et al.'s ROI/flow codec reproduction.
"""
import torch
from torch import nn
import torch.nn.functional as F
from .channel import ComplexChannel


class RGBLink(nn.Module):
    def __init__(self,complex_width=40,snr_db=10.,channel='awgn',pilots=8,
                 train_snr_min=-5.,train_snr_max=20.):
        super().__init__()
        if complex_width<1 or train_snr_max<train_snr_min:
            raise ValueError('invalid width or SNR range')
        self.width = complex_width
        self.snr_db = snr_db
        self.train_snr_min,self.train_snr_max = train_snr_min,train_snr_max
        channels = [6,64,96,128,128]
        self.encoder = nn.Sequential(*[layer for i in range(4) for layer in
            (nn.Conv2d(channels[i],channels[i+1],5,stride=2,padding=2),nn.GELU())],
            nn.Conv2d(128,2*complex_width,3,padding=1))
        self.decoder_input = nn.Sequential(nn.Conv2d(2*complex_width,128,3,padding=1),nn.GELU())
        self.decoder_stages = nn.ModuleList([nn.Sequential(nn.Conv2d(a,b,5,padding=2),nn.GELU())
                                             for a,b in [(128,128),(128,96),(96,64),(64,32)]])
        self.decoder_output = nn.Conv2d(32,6,3,padding=1)
        self.channel = ComplexChannel(channel,pilots)
        self.last_accounting = None

    def forward(self,left,right,batch_dict):
        if left.shape!=right.shape or left.ndim!=4 or left.shape[1]!=3:
            raise ValueError('matching Bx3xHxW stereo inputs required')
        code = self.encoder(torch.cat((left,right),1))
        packed = code.flatten(2).transpose(1,2).contiguous()
        packed = packed/(packed.square().sum(-1,keepdim=True)/self.width).clamp_min(1e-12).sqrt()
        snr = self.snr_db
        if self.training:
            snr = float(torch.empty(()).uniform_(self.train_snr_min,self.train_snr_max))
        received,accounting = self.channel(packed.reshape(left.shape[0],-1,2),snr)
        recovered = received.reshape_as(packed).transpose(1,2).reshape_as(code)
        image = self.decoder_input(recovered)
        for layer in self.decoder_stages:
            image = layer(F.interpolate(image,scale_factor=2,mode='bilinear',align_corners=False))
        image = self.decoder_output(image)
        if image.shape[-2:]!=left.shape[-2:]:
            image = F.interpolate(image,size=left.shape[-2:],mode='bilinear',align_corners=False)
        accounting.update(allocation='uniform_RGB_comparator',
            cbr_complex_per_input_real_scalar=accounting['total_complex_uses']/(6*left.shape[-2]*left.shape[-1]))
        self.last_accounting = accounting
        batch_dict['communication_accounting'] = accounting
        return image[:,:3],image[:,3:]
