"""Source-only stages for true rate-dependent padding, no wireless call."""
import importlib.util
import math
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'reproduction/cao2025'))
from radio import checked_boxes
from stages import charbonnier

spec = importlib.util.spec_from_file_location('original_source_rates005',
    ROOT/'experiments/original-paper-scenarios/original-source-rate-code-005/model.py')
architecture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(architecture)
SemanticRateVariant = architecture.SemanticRateVariant
FACTORS = architecture.FACTORS


def masks(shape, boxes, like, global_factor, key_factor):
    checked_boxes(shape, boxes)
    h, w = shape
    factor = math.lcm(global_factor, key_factor)
    native = []
    for view in boxes:
        value = like.new_zeros((1, 1, h, w))
        for box in view:
            x1, y1, x2, y2 = box['xyxy']
            value[..., y1:y2, x1:x2] = 1
        native.append(value)
    padded = tuple(F.pad(value, (0, (-w)%factor, 0, (-h)%factor)) for value in native)
    return tuple(native), padded


def forward_loss(model, left, right, boxes, current):
    stage = current['stage']
    if stage not in (1, 2, 3, 4):
        raise ValueError('source stages only; stage5 needs a separate radio contract')
    semantic = model.semantic
    h, w = left.shape[-2:]
    native, padded = masks((h, w), boxes, left, semantic.global_factor, semantic.key_factor)
    targets = (left, right)
    if stage == 1:
        outputs = semantic.global_warmup(left, right)
    elif stage == 2:
        pl, pr, _ = semantic.padded_images(left, right)
        outputs = tuple((decoder(encoder(image*mask))*mask)[..., :h, :w]
            for encoder, decoder, image, mask in zip(semantic.key_encoders,
                semantic.key_decoders, (pl, pr), padded))
        targets = tuple(image*mask for image, mask in zip(targets, native))
    else:
        outputs = semantic(left, right, native)
    loss = torch.stack([charbonnier(a, b) for a, b in zip(outputs, targets)]).mean()
    if current['loss'] == 'hybrid_charbonnier':
        masked = torch.stack([charbonnier(a*mask, b*mask)
            for a, b, mask in zip(outputs, targets, native)]).mean()
        loss = masked + .5*loss
    return dict(loss=loss, outputs=outputs, erasure=None, accounting=None)
