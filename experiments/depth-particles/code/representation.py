"""Bounded three equal-mass depth particles and two charged complex symbols."""
import numpy as np
import torch


def particles(probability, edges):
    assert probability.ndim >= 1 and edges.ndim == 1
    assert probability.shape[-1] + 1 == len(edges)
    assert torch.isfinite(probability).all() and (probability >= 0).all()
    assert torch.isfinite(edges).all() and (edges[1:] > edges[:-1]).all()
    total = probability.sum(-1, keepdim=True); assert (total > 0).all()
    mass = probability / total; cumulative = mass.cumsum(-1)
    levels = probability.new_tensor([1/6, 1/2, 5/6]).expand(probability.shape[:-1] + (3,)).contiguous()
    index = torch.searchsorted(cumulative.contiguous(), levels).clamp_max(probability.shape[-1] - 1)
    previous = torch.cat((torch.zeros_like(cumulative[..., :1]), cumulative[..., :-1]), -1)
    selected = mass.gather(-1, index); assert (selected > 0).all()
    within = (levels - previous.gather(-1, index)) / selected
    return edges[index] + within * (edges[index + 1] - edges[index])


def scale_depth(value, lower, upper):
    assert upper > lower
    return 2 * (value - lower) / (upper - lower) - 1


def unscale_depth(value, lower, upper):
    assert upper > lower
    return lower + (value + 1) * ((upper - lower) / 2)


def project_ordered_cube(value):
    """Exact three-coordinate L2 projection by contiguous block partitions."""
    assert value.shape[-1] == 3 and torch.isfinite(value).all()
    a, b, d = value.unbind(-1); ab = (a + b) / 2; bd = (b + d) / 2; mean = value.mean(-1)
    candidates = torch.stack((value, torch.stack((ab, ab, d), -1),
                              torch.stack((a, bd, bd), -1), torch.stack((mean, mean, mean), -1)), -2).clamp(-1, 1)
    valid = (candidates[..., 1:] >= candidates[..., :-1]).all(-1)
    distance = (candidates - value.unsqueeze(-2)).square().sum(-1).masked_fill(~valid, float('inf'))
    index = distance.argmin(-1, keepdim=True).unsqueeze(-1).expand(value.shape[:-1] + (1, 3))
    return candidates.gather(-2, index).squeeze(-2)


def encode(value):
    assert value.shape[-1] == 3 and torch.isfinite(value).all()
    assert ((value >= -1) & (value <= 1)).all() and (value[..., 1:] >= value[..., :-1]).all()
    anchor = torch.ones_like(value[..., :1])
    return torch.cat((value, anchor), -1) * (2 / (1 + value.square().sum(-1, keepdim=True))).sqrt()


def decode(received):
    """Received-only fixed-anchor floor and exact ordered bounded projection."""
    assert received.shape[-1] == 4 and torch.isfinite(received).all()
    anchor = received[..., 3:].clamp_min(2**-.5)
    return project_ordered_cube(received[..., :3] / anchor)


def histogram_W1(probability, edges, atoms):
    """Exact piecewise-uniform histogram to three equal-mass ordered atoms."""
    probability = np.asarray(probability, dtype=np.float64); edges = np.asarray(edges, dtype=np.float64)
    atoms = np.asarray(atoms, dtype=np.float64)
    assert probability.shape[-1] + 1 == len(edges) and atoms.shape[-1] == 3
    assert np.isfinite(probability).all() and (probability >= 0).all() and (probability.sum(-1) > 0).all()
    assert np.isfinite(atoms).all() and (np.diff(atoms, axis=-1) >= 0).all()
    assert np.isfinite(edges).all() and (np.diff(edges) > 0).all()
    mass = probability / probability.sum(-1, keepdims=True)
    end = mass.cumsum(-1)[..., :, None]; start = end - mass[..., :, None]
    lo = np.maximum(start, np.arange(3) / 3); hi = np.minimum(end, (np.arange(3) + 1) / 3)
    width = np.maximum(hi - lo, 0)
    divisor = np.where(mass > 0, mass, 1)[..., :, None]
    first = edges[:-1, None] + (lo - start) / divisor * np.diff(edges)[:, None] - atoms[..., None, :]
    last = edges[:-1, None] + (hi - start) / divisor * np.diff(edges)[:, None] - atoms[..., None, :]
    total_abs = np.abs(first) + np.abs(last)
    trapezoid = total_abs / 2
    crossing = (first**2 + last**2) / (2 * np.where(total_abs > 0, total_abs, 1))
    area = width * np.where(first * last >= 0, trapezoid, crossing)
    return area.sum(axis=(-2, -1))
