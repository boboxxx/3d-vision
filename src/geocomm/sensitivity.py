"""Training-only first-order task gradients distilled into sender importance."""
import math
import torch
from torch import nn
import torch.nn.functional as F


def normalized_log_importance(logits):
    """Log of positive weights with per-frame mean one; no global offset."""
    return logits - logits.logsumexp(1, keepdim=True) + math.log(logits.shape[1])


class TaskSensitivity(nn.Module):
    def __init__(self, cost_channels, appearance_channels, hidden=16):
        super().__init__()
        self.cost = nn.Sequential(nn.Conv3d(cost_channels, hidden, 1), nn.GELU(),
                                  nn.Conv3d(hidden, 1, 1))
        self.appearance = nn.Sequential(nn.Conv2d(appearance_channels, hidden, 1), nn.GELU(),
                                        nn.Conv2d(hidden, 1, 1))

    def forward(self, pooled_cost, pooled_appearance):
        # Auxiliary supervision fits the predictor without changing the encoder.
        cost = self.cost(pooled_cost.detach()).flatten(1)
        appearance = self.appearance(pooled_appearance.detach()).flatten(1)
        return normalized_log_importance(torch.cat((cost, appearance), 1).clamp(-8, 8))

    @staticmethod
    def distillation_loss(detection_loss, received, predicted_log_importance):
        """Target is ||d L_3D / d received_symbols||² per transmitted location.

        L_3D contains native classification and box regression/direction terms,
        excluding LiDAR imitation, depth and 2D auxiliaries. No second derivative.
        """
        gradient, = torch.autograd.grad(detection_loss, received, retain_graph=True,
                                       create_graph=False)
        if not torch.isfinite(gradient).all():
            raise RuntimeError('nonfinite native detection sensitivity')
        locations = predicted_log_importance.shape[1]
        target = gradient.detach().reshape(received.shape[0], locations, -1).square().sum(-1)
        target = target.clamp_min(1e-12)
        target_log = (target / target.mean(1, keepdim=True)).log()
        return F.smooth_l1_loss(predicted_log_importance, target_log), target_log
