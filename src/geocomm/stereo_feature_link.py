"""F5b native full-resolution fixed-layout feature transport before receiver-side stereo construction."""
import torch
from torch import nn
import torch.nn.functional as F
from .channel import ComplexChannel
from .link import GridPool


class StereoFeatureLink(nn.Module):
    def __init__(self,stereo_channels=32,appearance_channels=32,channel='awgn',snr_db=10.,pilots=8):
        super().__init__()
        if stereo_channels!=32 or appearance_channels!=32:
            raise ValueError('F5 locked native32-channel interfaces required')
        self.stereo_encoder=nn.Sequential(GridPool((16,1)),nn.Conv2d(32,32,1),nn.GELU(),nn.Conv2d(32,2,1))
        self.stereo_decoder=nn.Sequential(nn.Conv2d(2,32,3,padding=1),nn.GELU(),nn.Conv2d(32,32,3,padding=1))
        self.appearance_encoder=nn.Sequential(GridPool((4,2)),nn.Conv2d(32,32,1),nn.GELU(),nn.Conv2d(32,8,1))
        self.appearance_decoder=nn.Sequential(nn.Conv2d(8,32,3,padding=1),nn.GELU(),nn.Conv2d(32,32,3,padding=1))
        self.channel=ComplexChannel(channel,pilots);self.snr_db=float(snr_db);self.last_accounting=None

    @staticmethod
    def layout(feature_shapes):
        if len(feature_shapes)!=3: raise ValueError('three public native feature shapes required')
        left,right,appearance=feature_shapes
        b,c,h,w=left
        if b!=1 or c!=32 or min(h,w)<4 or h%4 or w%4 or tuple(right)!=tuple(left) or tuple(appearance)!=(b,c,h//4,w//4):
            raise ValueError('native full-resolution stereo/quarter-resolution appearance layout required')
        return [(b,2,(h+15)//16,w)]*2+[(b,8,(h//4+3)//4,(w//4+1)//2)]

    @staticmethod
    def pack(code): return code.flatten(2).transpose(1,2).contiguous().reshape(code.shape[0],-1,2)

    def decode(self,received,public_feature_shapes):
        """Only received symbols and public fixed model geometry enter decoding."""
        layout=self.layout(public_feature_shapes);counts=[c*h*w//2 for _,c,h,w in layout]
        if received.shape!=(public_feature_shapes[0][0],sum(counts),2): raise ValueError('received duration/layout differs')
        restored=[]
        for i,(stream,shape) in enumerate(zip(received.split(counts,dim=1),layout)):
            b,c,h,w=shape;code=stream.reshape(b,h*w,c).transpose(1,2).reshape(shape)
            decoded=(self.stereo_decoder if i<2 else self.appearance_decoder)(code)
            restored.append(F.interpolate(decoded,size=public_feature_shapes[i][2:],mode='bilinear',align_corners=False))
        return tuple(restored)

    def forward(self,left_stereo,right_stereo,appearance,batch):
        public_shapes=tuple(tuple(feature.shape) for feature in (left_stereo,right_stereo,appearance))
        expected=self.layout(public_shapes)
        codes=(self.stereo_encoder(left_stereo),self.stereo_encoder(right_stereo),self.appearance_encoder(appearance))
        if [tuple(code.shape) for code in codes]!=expected: raise RuntimeError('encoded fixed layout differs')
        symbols=torch.cat([self.pack(code) for code in codes],dim=1)
        energy=symbols.square().sum(-1).mean(1,keepdim=True)
        if not torch.isfinite(energy).all() or (energy<=0).any(): raise ValueError('invalid transmitted code energy')
        symbols=symbols/energy.sqrt().unsqueeze(-1)
        received,account=self.channel(symbols,self.snr_db)
        outputs=self.decode(received,public_shapes)
        pixels=batch['left_img'].shape[-2]*batch['left_img'].shape[-1]
        account.update(allocation='uniform',cbr_complex_per_input_real_scalar=account['total_complex_uses']/(6*pixels),
            boundary='stereo_features_before_receiver_cost',stereo_complex_uses=expected[0][1]*expected[0][2]*expected[0][3],
            appearance_complex_uses=expected[2][1]*expected[2][2]*expected[2][3]//2)
        self.last_accounting=account;batch['communication_accounting']=account
        return outputs
