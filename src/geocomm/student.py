"""Optional sensor encoder; the original feature extractor is a training teacher.

Produces the original full-resolution stereo and quarter-resolution appearance
feature interfaces. It changes sensor computation, not receiver depth bins,
projection, detector heads, or the communication resource definition.
"""
import torch
from torch import nn
import torch.nn.functional as F


class SeparableResidual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1,
            groups=channels, bias=False), nn.GroupNorm(8, channels), nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False), nn.GroupNorm(8, channels))

    def forward(self, x):
        return F.gelu(x + self.body(x))


class StereoStudentEncoder(nn.Module):
    def __init__(self, stereo_channels=32, appearance_channels=32):
        super().__init__()
        self.context = nn.Sequential(
            nn.Conv2d(3,16,3,stride=2,padding=1,bias=False),nn.GroupNorm(4,16),nn.GELU(),
            nn.Conv2d(16,32,3,stride=2,padding=1,bias=False),nn.GroupNorm(8,32),nn.GELU(),
            *[SeparableResidual(32) for _ in range(3)])
        # Full-resolution local evidence avoids discarding fine stereo edges.
        self.local = nn.Sequential(nn.Conv2d(3,8,3,padding=1,bias=False),
                                   nn.GroupNorm(4,8),nn.GELU())
        self.stereo_context = nn.Conv2d(32,24,1)
        self.stereo = nn.Sequential(nn.Conv2d(32,stereo_channels,1,bias=False),
            nn.Conv2d(stereo_channels,stereo_channels,3,padding=1,
                      groups=stereo_channels,bias=False),
            nn.Conv2d(stereo_channels,stereo_channels,1,bias=False))
        self.appearance = nn.Sequential(nn.Conv2d(32,appearance_channels,1,bias=False),
            nn.GroupNorm(8,appearance_channels),nn.GELU())

    def forward(self, image):
        if image.ndim!=4 or image.shape[1]!=3 or any(v%4 for v in image.shape[-2:]):
            raise ValueError('student requires Bx3xHxW with H and W divisible by four')
        context = self.context(image)
        full_context = F.interpolate(self.stereo_context(context),size=image.shape[-2:],
                                     mode='bilinear',align_corners=False)
        stereo = self.stereo(torch.cat((self.local(image),full_context),dim=1))
        return stereo,self.appearance(context)


def teacher_feature_targets(backbone,neck,left,right):
    """Frozen teacher inference without modifying BN statistics or mode state."""
    modes = [(module,module.training) for root in [backbone,neck] for module in root.modules()]
    try:
        backbone.eval()
        neck.eval()
        with torch.no_grad():
            left_targets = neck([left]+list(backbone(left)))
            right_targets = neck([right]+list(backbone(right)))
        return left_targets[0],right_targets[0],left_targets[1]
    finally:
        # Restore individual modes, including intentionally frozen nested BNs.
        for module,training in modes:
            module.training = training


def feature_distillation(students,targets):
    """Equal weighting of three interfaces, scaled by detached target energy."""
    if len(students)!=3 or len(targets)!=3:
        raise ValueError('left/right stereo and left appearance targets required')
    terms = []
    for student,target in zip(students,targets):
        if student.shape!=target.shape:
            raise ValueError('student/teacher feature interface differs')
        target = target.detach()
        terms.append(F.mse_loss(student,target)/target.square().mean().clamp_min(1e-6))
    return sum(terms)/len(terms)
