"""Received-only finite-grid node posterior; public prior is not KITTI calibration."""
import math
import torch
import torch.nn.functional as F


def likelihood(received, noise_power, grid_unit, fading=None):
    """ZF observations [...,3,2]; fading is actual perfect receiver CSI."""
    if received.shape[-2:] != (3, 2) or received.dtype not in (torch.float32, torch.float64):
        raise ValueError('three complex position observations required')
    if not math.isfinite(noise_power) or noise_power <= 0:
        raise ValueError('positive public noise power required')
    if not torch.isfinite(received).all():
        raise ValueError('nonfinite receiver observation')
    r = received.double()
    grid = grid_unit.to(device=r.device, dtype=torch.float64)
    if grid.ndim != 1 or len(grid) < 3 or not torch.isfinite(grid).all():
        raise ValueError('public one-dimensional grid required')
    if not bool((grid[1:] > grid[:-1]).all()) or float(grid[0]) != 0 or float(grid[-1]) != 1:
        raise ValueError('strict finite grid must span0..1')
    if fading is None:
        precision = r.new_ones(r.shape[:-1])/noise_power
    else:
        if fading.shape != received.shape or not torch.isfinite(fading).all():
            raise ValueError('actual complex receiver CSI must match position slots')
        energy = fading.double().square().sum(-1)
        if not bool((energy > 0).all()):
            raise ValueError('zero CSI cannot identify a ZF observation; no fade clipping')
        precision = energy/noise_power
    angle = (grid*2-1)*(math.pi/2)
    return 2*precision[..., None]*(r[..., 0, None]*angle.cos()+r[..., 1, None]*angle.sin())


def marginals(log_likelihood, ordered=True):
    """Exact discrete uniform prior over a<=b<=c, including all grid ties."""
    if log_likelihood.shape[-2] != 3 or not torch.isfinite(log_likelihood).all():
        raise ValueError('finite three-slot likelihood required')
    if not ordered:
        return log_likelihood.softmax(-1)
    l1, l2, l3 = log_likelihood.unbind(-2)
    a2 = l2+torch.logcumsumexp(l1, -1)
    a3 = l3+torch.logcumsumexp(a2, -1)
    z = a3.logsumexp(-1, keepdim=True)
    b2 = torch.logcumsumexp(l3.flip(-1), -1).flip(-1)
    b1 = torch.logcumsumexp((l2+b2).flip(-1), -1).flip(-1)
    return torch.stack(((l1+b1-z).exp(), (a2+b2-z).exp(), (a3-z).exp()), -2)


def fading_blind_likelihood(received, noise_power, grid_unit):
    """Marginal CN noise / CN fading likelihood; intentionally ignores CSI."""
    r = received.double()
    angle = (grid_unit.to(device=r.device, dtype=torch.float64)*2-1)*(math.pi/2)
    squared = (r[..., 0, None]-angle.cos()).square()+(r[..., 1, None]-angle.sin()).square()
    return -2*(noise_power+squared).log()


def isotonic_three(values, weights):
    """Fixed weighted L2 order-projection control, not a phase likelihood."""
    if values.shape != weights.shape or values.shape[-1] != 3:
        raise ValueError('three positive-weight coordinates required')
    if not torch.isfinite(values).all() or not torch.isfinite(weights).all() or not bool((weights > 0).all()):
        raise ValueError('finite positive weights required')
    x1, x2, x3 = values.unbind(-1)
    w1, w2, w3 = weights.unbind(-1)
    m12 = (w1*x1+w2*x2)/(w1+w2)
    m23 = (w2*x2+w3*x3)/(w2+w3)
    allmean = (w1*x1+w2*x2+w3*x3)/(w1+w2+w3)
    merge12 = x1 > x2
    merge23 = (~merge12) & (x2 > x3)
    mergeall = (merge12 & (m12 > x3)) | (merge23 & (x1 > m23))
    result = torch.stack((torch.where(merge12, m12, x1),
        torch.where(merge12, m12, torch.where(merge23, m23, x2)),
        torch.where(merge23, m23, x3)), -1)
    return torch.where(mergeall[..., None], allmean[..., None], result)


def decode(codec, received, cost_hw, appearance_hw, channel, snr_db,
           receiver_csi=None, mode='ordered', grid_points=289):
    """No TX scales, clean means/probabilities/features, noise samples or GT."""
    if codec.arm not in ('G', 'P', 'S') or codec.k != 3 or mode not in ('ordered', 'independent', 'isotonic', 'ordered_blind'):
        raise ValueError('three-node geometry codec and declared receiver mode required')
    if channel == 'identity':
        return codec.decode(received, cost_hw, appearance_hw)
    if channel not in ('awgn', 'rayleigh') or not math.isfinite(float(snr_db)):
        raise ValueError('declared AWGN or perfect-CSI Rayleigh required')
    if (channel == 'rayleigh') != (receiver_csi is not None):
        raise ValueError('perfect CSI supplied exactly for Rayleigh')
    if receiver_csi is not None and receiver_csi.shape != received.shape:
        raise ValueError('receiver CSI layout differs')
    if grid_points != 289:
        raise ValueError('production grid fixed before AP; no validation selection')
    # Reuse frozen sphere projection, feature decoding and interpolation only.
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[3]/'experiments/cost-field/code_004/codec.py'
    spec = importlib.util.spec_from_file_location('frozen_cost_codec004', path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    b = received.shape[0]
    h, w = cost_hw[0]//codec.stride, cost_hw[1]//codec.stride
    if received.shape != (b, h*w*19, 2):
        raise ValueError('same charged19 complex symbols per pooled site required')
    payload = received.reshape(b, h, w, 19, 2)
    csi = None if receiver_csi is None else receiver_csi.reshape(b, h, w, 19, 2)[..., :3, :]
    q = 10**(-float(snr_db)/10)
    grid = torch.linspace(0, 1, grid_points, device=received.device, dtype=torch.float64)
    lo, hi = codec.depth_axis[0].double(), codec.depth_axis[-1].double()
    axis = codec.depth_axis.double()
    sigma = codec.log_bandwidth.double().exp().clamp_min(torch.finfo(received.dtype).eps)
    if mode == 'isotonic':
        position = base.receive_project(payload[..., :3, :].unsqueeze(-2), 1).squeeze(-2)
        unit = (torch.atan2(position[..., 1].double(), position[..., 0].double()).clamp(-math.pi/2, math.pi/2)/(math.pi/2)+1)/2
        precision = torch.ones_like(unit) if csi is None else csi.double().square().sum(-1)
        mu = lo+(hi-lo)*isotonic_three(unit, precision)
        delta = axis[None, None, :, None, None]-mu.permute(0, 3, 1, 2)[:, :, None]
        weight = (-.5*(delta/sigma).square()).exp()/3
    else:
        ll = (fading_blind_likelihood(payload[..., :3, :], q, grid)
              if mode == 'ordered_blind' and channel == 'rayleigh'
              else likelihood(payload[..., :3, :], q, grid, csi))
        posterior = marginals(ll, ordered=mode != 'independent')
        centers = lo+(hi-lo)*grid
        basis = (-.5*((axis[:, None]-centers[None, :])/sigma).square()).exp()/3
        weight = torch.einsum('bhwkg,dg->bkdhw', posterior, basis)
    value = base.receive_project(payload[..., 3:15, :].reshape(b, h, w, 3, 4, 2), 4).reshape(b, h, w, 3, 8)
    value = value.permute(0, 3, 4, 1, 2).reshape(b*3, 8, h, w)
    coefficients = codec.cost_decoder(value).reshape(b, 3, 32, h, w)
    cost = torch.einsum('bkdhw,bkchw->bcdhw', weight.to(coefficients.dtype), coefficients)
    cost = F.interpolate(cost, size=(len(codec.depth_axis), *cost_hw), mode='trilinear', align_corners=True)
    app = base.receive_project(payload[..., 15:, :], 4).reshape(b, h, w, 8).permute(0, 3, 1, 2)
    app = F.interpolate(codec.appearance_decoder(app), size=appearance_hw, mode='bilinear', align_corners=True)
    return cost, app
