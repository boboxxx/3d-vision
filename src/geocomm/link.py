"""Dense fixed-layout cost-volume and appearance latent JSCC link."""
import torch
from torch import nn
import torch.nn.functional as F
from .allocation import allocate_power, depth_moments
from .channel import ComplexChannel


class GridPool(nn.Module):
    """Pool to ceil-sized grids, including inputs smaller than a stride."""
    def __init__(self, strides):
        super().__init__()
        self.strides = tuple(strides)
        if any(int(s) != s or s < 1 for s in self.strides):
            raise ValueError("pool strides must be positive integers")

    def forward(self, x):
        shape = [(size + stride - 1) // stride
                 for size, stride in zip(x.shape[2:], self.strides)]
        pool = F.adaptive_avg_pool3d if len(self.strides) == 3 else F.adaptive_avg_pool2d
        return pool(x, shape)


class GeometryLink(nn.Module):
    def __init__(self, cost_channels=32, appearance_channels=32,
                 complex_width=4, snr_db=10., channel="awgn", pilots=8,
                 allocation="geometry", posterior_weight=0.1,
                 posterior_hidden=0,
                 task_sensitivity=False, sensitivity_weight=0.1,
                 train_snr_min=-5., train_snr_max=20.,
                 geometry_stride=(8, 4, 4), appearance_stride=4):
        super().__init__()
        if allocation not in ("geometry", "geometry_task", "task", "uniform", "learned"):
            raise ValueError("unknown allocation")
        if allocation in ('geometry_task', 'task') and not task_sensitivity:
            raise ValueError('task allocation requires sensitivity predictor')
        if sensitivity_weight <= 0:
            raise ValueError('positive sensitivity supervision weight required')
        if complex_width < 1 or train_snr_max < train_snr_min:
            raise ValueError("invalid rate or SNR interval")
        self.width = complex_width
        self.snr_db = snr_db
        self.allocation = allocation
        self.posterior_weight = posterior_weight
        self.sensitivity_weight = sensitivity_weight
        from .sensitivity import TaskSensitivity
        self.sensitivity_predictor = TaskSensitivity(cost_channels, appearance_channels) if task_sensitivity else None
        self.train_snr_min, self.train_snr_max = train_snr_min, train_snr_max
        if int(posterior_hidden)!=posterior_hidden or posterior_hidden<0:
            raise ValueError('nonnegative integer posterior hidden width required')
        # A linear head on raw concat cost cancels depth-constant left features
        # in softmax. Nonlinear coupling is required at the raw-cost boundary.
        self.posterior = nn.Sequential(nn.Conv3d(cost_channels,posterior_hidden,1),
            nn.GELU(),nn.Conv3d(posterior_hidden,1,1)) if posterior_hidden else nn.Conv3d(cost_channels,1,1)
        self.generic_score = nn.Conv3d(cost_channels, 1, 1)
        self.cost_encoder = nn.Sequential(
            GridPool(geometry_stride),
            nn.Conv3d(cost_channels, cost_channels, 3, padding=1),
            nn.GELU(), nn.Conv3d(cost_channels, 2 * complex_width, 1))
        self.cost_decoder = nn.Sequential(
            nn.Conv3d(2 * complex_width, cost_channels, 3, padding=1),
            nn.GELU(), nn.Conv3d(cost_channels, cost_channels, 3, padding=1))
        self.appearance_encoder = nn.Sequential(
            GridPool((appearance_stride, appearance_stride)),
            nn.Conv2d(appearance_channels, appearance_channels, 3, padding=1),
            nn.GELU(), nn.Conv2d(appearance_channels, 2 * complex_width, 1))
        self.appearance_decoder = nn.Sequential(
            nn.Conv2d(2 * complex_width, appearance_channels, 3, padding=1),
            nn.GELU(), nn.Conv2d(appearance_channels, appearance_channels, 3, padding=1))
        self.channel = ComplexChannel(channel, pilots)
        self.last_accounting = None

    @staticmethod
    def _pack(code):
        return code.flatten(2).transpose(1, 2).contiguous()

    @staticmethod
    def _unpack(tokens, shape):
        return tokens.transpose(1, 2).reshape(shape)

    def _posterior_loss(self, logits, depth_samples, depth_gt):
        # Interpolate logits to measured depth locations; do not average sparse
        # zero-filled LiDAR images into false depth supervision.
        if depth_gt.ndim == 4:
            depth_gt = depth_gt[:, 0]
        z = torch.as_tensor(depth_samples, device=logits.device, dtype=logits.dtype)
        # Samples are bin centers. Include measured depths in the edge half-bins;
        # zero-filled missing LiDAR pixels must never become supervision.
        lower=z[0]-(z[1]-z[0])/2 if z.numel()>1 else z[0]
        upper=z[-1]+(z[-1]-z[-2])/2 if z.numel()>1 else z[-1]
        valid = torch.isfinite(depth_gt) & (depth_gt>0) & (depth_gt>=lower) & (depth_gt<=upper)
        if not valid.any():
            return logits.sum() * 0
        up = F.interpolate(logits, size=depth_gt.shape[-2:], mode="bilinear", align_corners=False)
        measured_logits = up.permute(0, 2, 3, 1)[valid]
        target = (depth_gt[valid][:, None] - z[None]).abs().argmin(-1)
        return F.cross_entropy(measured_logits, target)

    def forward(self, cost, appearance, batch_dict, depth_samples):
        if cost.shape[0] != appearance.shape[0]:
            raise ValueError("cost/appearance batch mismatch")
        logits = self.posterior(cost).squeeze(1)
        probability = logits.softmax(1)
        _, risk = depth_moments(probability, depth_samples)
        cost_code = self.cost_encoder(cost)
        app_code = self.appearance_encoder(appearance)
        c = self._pack(cost_code)
        a = self._pack(app_code)
        tokens = torch.cat((c, a), 1)
        predicted_importance = None
        if self.sensitivity_predictor is not None:
            predicted_importance = self.sensitivity_predictor(
                self.cost_encoder[0](cost), self.appearance_encoder[0](appearance))
        snr = self.snr_db
        if self.training:
            snr = float(torch.empty(()).uniform_(self.train_snr_min, self.train_snr_max))
        if self.allocation == "learned":
            # Generic importance is learned through differentiable power scaling.
            score = self.generic_score(cost).mean(2)
            rc = F.adaptive_avg_pool2d(score, cost_code.shape[-2:]).flatten(1)
            rc = rc[:, None].expand(-1, cost_code.shape[2], -1).reshape(cost.shape[0], -1)
            ra = F.adaptive_avg_pool2d(score, app_code.shape[-2:]).flatten(1)
            power = torch.cat((rc, ra), 1).softmax(1) * tokens.shape[1]
        elif self.allocation in ("geometry", "geometry_task", "task"):
            rc = F.adaptive_avg_pool2d(risk, cost_code.shape[-2:]).flatten(1)
            rc = rc[:, None].expand(-1, cost_code.shape[2], -1).reshape(cost.shape[0], -1)
            ra = F.adaptive_avg_pool2d(risk, app_code.shape[-2:]).flatten(1)
            weights = torch.cat((rc, ra), 1)
            if self.allocation == 'geometry_task':
                weights = weights * predicted_importance.detach().exp()
            elif self.allocation == 'task':
                weights = predicted_importance.detach().exp()
            power = allocate_power(weights, 10 ** (snr / 10))
        else:
            power = torch.ones(tokens.shape[:2], dtype=tokens.dtype, device=tokens.device)
        # Per-location average complex energy = 1 before resource allocation.
        tokens = tokens / (tokens.square().sum(-1, keepdim=True) / self.width).clamp_min(1e-12).sqrt()
        tokens = tokens * power[..., None].sqrt()
        received, accounting = self.channel(tokens.reshape(cost.shape[0], -1, 2), snr)
        recovered = received.reshape_as(tokens)
        if self.training and predicted_importance is not None:
            batch_dict['communication_sensitivity_context'] = (received, predicted_importance)
        clean_cost = self.cost_decoder(self._unpack(recovered[:, :c.shape[1]], cost_code.shape))
        clean_app = self.appearance_decoder(self._unpack(recovered[:, c.shape[1]:], app_code.shape))
        clean_cost = F.interpolate(clean_cost, size=cost.shape[2:], mode="trilinear", align_corners=False)
        clean_app = F.interpolate(clean_app, size=appearance.shape[2:], mode="bilinear", align_corners=False)
        pixels = batch_dict["left_img"].shape[-2] * batch_dict["left_img"].shape[-1]
        accounting["cbr_complex_per_input_real_scalar"] = accounting["total_complex_uses"] / (6 * pixels)
        accounting["allocation"] = self.allocation
        self.last_accounting = accounting
        batch_dict["communication_accounting"] = accounting
        if self.training:
            if "depth_gt_img" not in batch_dict:
                raise ValueError("posterior training requires measured depth_gt_img")
            batch_dict["communication_aux_loss"] = self.posterior_weight * self._posterior_loss(
                logits, depth_samples, batch_dict["depth_gt_img"])
        return clean_cost, clean_app
