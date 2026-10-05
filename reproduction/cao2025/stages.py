"""Verified five-stage schedule plus explicitly declared loss-reduction choices."""
import torch
from radio import roi_masks

EPOCHS = {1: 12, 2: 10, 3: 6, 4: 10, 5: 45}
COMPONENTS = ('global_encoder', 'global_decoder', 'flow', 'key_encoders',
              'key_decoders', 'fusions', 'global_channel', 'key_channel')


def component(name):
    if name.startswith('semantic.global_decoder.flow.'):
        return 'flow'
    for key in COMPONENTS:
        prefix = key+'.' if key.endswith('_channel') else 'semantic.'+key+'.'
        if name.startswith(prefix):
            return key
    raise ValueError('unmapped parameter/state component: '+name)


def phase(stage, epoch):
    if stage not in EPOCHS or not 1 <= epoch <= EPOCHS[stage]:
        raise ValueError('invalid one-based stage/epoch')
    if stage == 1:
        rates = dict(global_encoder=2e-4, global_decoder=2e-4, fusions=1e-4)
        if epoch > 2:
            rates['flow'] = 2.5e-5
        loss = 'global_charbonnier'
    elif stage == 2:
        rates = dict(key_encoders=1e-4, key_decoders=2e-4)
        loss = 'masked_charbonnier'
    elif stage == 3:
        rates = dict(fusions=1e-4)
        loss = 'hybrid_charbonnier'
    else:
        rates = dict(global_encoder=1e-4, global_decoder=1e-4, key_encoders=2e-5,
                     key_decoders=2e-5, fusions=2e-5, flow=1.25e-5)
        if stage == 5:
            rates.update(global_channel=1e-4, key_channel=1e-4)
            if epoch <= 25:
                rates = {key: rates[key] for key in ('global_channel', 'key_channel')}
            loss = 'semantic_mse' if epoch <= 25 else 'global_charbonnier'
        else:
            # The primary paper leaves the exact switch unspecified. This
            # fixed5/5 variant must be locked before training, not selected.
            loss = 'hybrid_charbonnier' if epoch <= 5 else 'global_charbonnier'
    return dict(stage=stage, epoch=epoch, rates=rates, loss=loss)


def configure(model, current):
    """Freeze inactive weights AND BatchNorm statistics at every phase switch."""
    model.eval()
    active = set(current['rates'])
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(component(name) in active)
    for key in active:
        if key.endswith('_channel'):
            getattr(model, key).train()
        elif key == 'flow':
            model.semantic.global_decoder.flow.train()
        else:
            getattr(model.semantic, key).train()
    # GlobalDecoder.train() recursively marks the flow trainable mode even
    # during the first two epochs; restore its explicitly frozen phase mode.
    if 'flow' not in active:
        model.semantic.global_decoder.flow.eval()
    return {name for name, parameter in model.named_parameters() if parameter.requires_grad}


def optimizer_groups(model, stage):
    # Include the stage's union from the beginning. Inactive parameters have
    # grad=None and no Adam state until unfreezing; earlier states survive the
    # phase switch. Construct a fresh optimizer between stages only.
    rates = phase(stage, EPOCHS[stage])['rates']
    groups = []
    for key in COMPONENTS:
        if key not in rates:
            continue
        names, parameters = [], []
        for name, parameter in model.named_parameters():
            if component(name) == key:
                names.append(name); parameters.append(parameter)
        if not names:
            raise ValueError('missing required component: '+key)
        groups.append(dict(params=parameters, lr=rates[key], component=key, parameter_names=names))
    return groups


def frozen_state_keys(model):
    active = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    # Only BatchNorm running statistics of currently active channel codecs may
    # change. SpyNet RGB mean/std always stay fixed, even when flow is trained.
    allowed_buffers = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.training:
            allowed_buffers.update(name+'.'+key for key in ('running_mean', 'running_var', 'num_batches_tracked'))
    return set(model.state_dict())-active-allowed_buffers


def charbonnier(a, b, epsilon=1e-3):
    # Variant: elementwise average over native pixels/channels, epsilon1e-3.
    return torch.sqrt((a-b).square()+epsilon**2).mean()


def forward_loss(model, left, right, boxes, current, snr_db=10.):
    h, w = left.shape[-2:]
    masks = roi_masks((h, w), boxes, left)
    native_masks = tuple(mask[..., :h, :w] for mask in masks)
    semantic = model.semantic
    stage = current['stage']
    if stage == 1:
        outputs = semantic.global_warmup(left, right)
        targets = (left, right)
    elif stage == 2:
        padded_left, padded_right, _ = semantic.padded_images(left, right)
        outputs = tuple((decoder(encoder(image*mask))*mask)[..., :h, :w]
            for encoder, decoder, image, mask in zip(semantic.key_encoders, semantic.key_decoders,
                (padded_left, padded_right), masks))
        targets = tuple(image*mask for image, mask in zip((left, right), native_masks))
    elif stage in (3, 4):
        outputs = semantic(left, right, native_masks)
        targets = (left, right)
    else:
        outcome = model(left, right, boxes, channel='awgn', snr_db=snr_db,
                        decode_rgb=current['loss']!='semantic_mse')
        if outcome['erasure'] is not None:
            return dict(loss=None, outputs=None, erasure=outcome['erasure'], accounting=outcome['accounting'])
        if current['loss']=='semantic_mse':
            return dict(loss=outcome['semantic_mse'], outputs=None, erasure=None, accounting=outcome['accounting'])
        outputs, targets = outcome['outputs'], (left, right)
    loss = torch.stack([charbonnier(a, b) for a, b in zip(outputs, targets)]).mean()
    if current['loss']=='hybrid_charbonnier':
        masked = torch.stack([charbonnier(a*mask, b*mask)
            for a, b, mask in zip(outputs, targets, native_masks)]).mean()
        loss = masked+.5*loss
    return dict(loss=loss, outputs=outputs, erasure=None,
                accounting=outcome['accounting'] if stage==5 else None)
