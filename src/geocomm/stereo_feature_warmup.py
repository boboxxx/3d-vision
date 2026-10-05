"""Exact F5 codec-only scope and capture before native stereo matching."""
import torch
from torch import nn
import torch.nn.functional as F
from .codec_warmup import assert_frozen

PREFIX='backbone_3d.stereo_feature_link.'
CONVOLUTIONS={'stereo_encoder':(1,3),'stereo_decoder':(0,2),
              'appearance_encoder':(1,3),'appearance_decoder':(0,2)}


def freeze_except_codec(model):
    expected={PREFIX+branch+'.'+str(index)+'.'+kind for branch,indices in CONVOLUTIONS.items()
              for index in indices for kind in ('weight','bias')}
    parameters=dict(model.named_parameters())
    if not expected<=set(parameters): raise ValueError('missing required F5 codec parameter')
    model.eval()
    for name,parameter in parameters.items(): parameter.requires_grad_(name in expected)
    link=model.backbone_3d.stereo_feature_link
    if model.backbone_3d.semantic_link is not None or link.channel.kind!='identity':
        raise ValueError('F5 requires stereo feature identity link only')
    for module in link.modules():
        if isinstance(module,(nn.modules.batchnorm._BatchNorm,nn.modules.dropout._DropoutNd)):
            raise ValueError('locked eval codec cannot contain train-dependent layers')
    return [(name,parameter) for name,parameter in parameters.items() if name in expected]


class FeatureLossCaptured(Exception):
    def __init__(self,loss,scalars):
        super().__init__('native forward stopped before receiver cost construction')
        self.loss,self.scalars=loss,scalars


def capture_reconstruction(backbone,sensor_batch):
    link=backbone.stereo_feature_link
    if backbone.training or link.training or backbone.semantic_link is not None or link.channel.kind!='identity':
        raise ValueError('F5 requires eval-mode stereo feature identity link')
    if set(sensor_batch)-{'batch_size','left_img','right_img','calib','image_shape','frame_id'}:
        raise ValueError('non-sensor input entered F5 feature forward')
    if sensor_batch['batch_size']!=1: raise ValueError('fixed batch1 required')
    def capture(module,inputs,outputs):
        targets=inputs[:3]
        if len(outputs)!=3 or any(target.requires_grad for target in targets):
            raise RuntimeError('clean frozen feature target scope differs')
        losses=[F.mse_loss(received,target.detach()) for received,target in zip(outputs,targets)]
        loss=.5*(losses[0]+losses[1])+losses[2]
        if not loss.requires_grad: raise RuntimeError('codec loss has no gradient')
        account=module.last_accounting
        scalars={branch+'_mse':float(value.detach()) for branch,value in zip(('left_stereo','right_stereo','appearance'),losses)}
        scalars.update({branch+'_shape':list(value.shape) for branch,value in zip(('left_stereo','right_stereo','appearance'),targets)})
        for key in ('channel','allocation','boundary','data_complex_uses','total_complex_uses','pilot_complex_uses',
                    'header_complex_uses','stereo_complex_uses','appearance_complex_uses','snr_db'):
            scalars[key]=account[key]
        scalars.update(tx_energy=float(account['tx_energy_per_frame'][0]),cbr=account['cbr_complex_per_input_real_scalar'])
        raise FeatureLossCaptured(loss,scalars)
    handle=link.register_forward_hook(capture)
    try: backbone(sensor_batch)
    except FeatureLossCaptured as captured: return captured.loss,captured.scalars
    finally: handle.remove()
    raise RuntimeError('native backbone never reached required F5 link boundary')
