"""F4 exact codec scope and native-backbone stop after reconstruction loss."""
import torch
from torch import nn
import torch.nn.functional as F

PREFIX = 'backbone_3d.semantic_link.'
CONVOLUTIONS = {'cost_encoder': (1, 3), 'cost_decoder': (0, 2),
                'appearance_encoder': (1, 3), 'appearance_decoder': (0, 2)}


def freeze_except_codec(model):
    expected = {PREFIX+branch+'.'+str(index)+'.'+kind
                for branch, indices in CONVOLUTIONS.items() for index in indices
                for kind in ('weight', 'bias')}
    parameters = dict(model.named_parameters())
    if not expected <= set(parameters):
        raise ValueError('missing required codec parameter')
    model.eval()
    for name, parameter in parameters.items():
        parameter.requires_grad_(name in expected)
    # Parent eval retains gradients and prevents posterior/sensitivity training.
    link = model.backbone_3d.semantic_link
    if link.channel.kind != 'identity' or link.allocation != 'uniform':
        raise ValueError('F4 reconstruction requires identity and uniform power')
    for branch in CONVOLUTIONS:
        for module in getattr(link, branch).modules():
            if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.modules.dropout._DropoutNd)):
                raise ValueError('eval-only codec warmup would freeze a train-dependent layer')
    return [(name, parameter) for name, parameter in parameters.items() if name in expected]


def assert_frozen(reference, actual, trainable_names):
    if set(reference) != set(actual):
        raise ValueError('full checkpoint state coverage differs')
    checked = 0
    for name, value in reference.items():
        if name in trainable_names:
            continue
        received = actual[name].detach().cpu()
        if value.shape != received.shape or value.dtype != received.dtype or not torch.equal(value, received):
            raise ValueError('inactive state changed: '+name)
        checked += 1
    return checked


class CodecLossCaptured(Exception):
    def __init__(self, loss, scalars):
        super().__init__('native forward stopped after codec reconstruction')
        self.loss, self.scalars = loss, scalars


def capture_reconstruction(backbone, sensor_batch):
    if backbone.training or backbone.semantic_link.training:
        raise ValueError('frozen backbone/link must remain in eval mode')
    link = backbone.semantic_link
    if backbone.semantic_link_boundary != 'raw_cost' or link.channel.kind != 'identity':
        raise ValueError('incorrect raw-cost/identity boundary')
    if set(sensor_batch)-{'batch_size','left_img','right_img','calib','image_shape','frame_id'}:
        raise ValueError('non-sensor input entered F4 feature forward')
    if sensor_batch['batch_size'] != 1:
        raise ValueError('fixed batch1 required')
    def capture(module, inputs, outputs):
        cost, appearance, _, _ = inputs
        if cost.requires_grad or appearance.requires_grad:
            raise RuntimeError('clean frozen sender target unexpectedly has gradients')
        losses = [F.mse_loss(recovered, target.detach()) for recovered, target in zip(outputs, (cost, appearance))]
        loss = losses[0]+losses[1]
        if not loss.requires_grad:
            raise RuntimeError('codec reconstruction loss has no gradient')
        account = module.last_accounting
        raise CodecLossCaptured(loss, dict(cost_mse=float(losses[0].detach()),
            appearance_mse=float(losses[1].detach()), cost_shape=list(cost.shape),
            appearance_shape=list(appearance.shape), channel=account['channel'],
            allocation=account['allocation'], data_complex_uses=account['data_complex_uses'],
            total_complex_uses=account['total_complex_uses'],
            pilot_complex_uses=account['pilot_complex_uses'],header_complex_uses=account['header_complex_uses'],
            snr_db=account['snr_db'],tx_energy=float(account['tx_energy_per_frame'][0]),
            cbr=account['cbr_complex_per_input_real_scalar']))
    handle = link.register_forward_hook(capture)
    try:
        backbone(sensor_batch)
    except CodecLossCaptured as captured:
        return captured.loss, captured.scalars
    finally:
        handle.remove()
    raise RuntimeError('native backbone never reached required codec boundary')
