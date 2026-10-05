"""Conditional Gaussian proxy allocation; not a task RD theorem."""
import torch


@torch.no_grad()
def allocate_power(risk, snr_linear, iterations=64):
    """Minimize sum a/(1+s*p), p>=0, sum p=N, independently per frame.

    Risk is a nonnegative [batch, locations] tensor. The multiplier is solved
    by bisection. Selection has no gradient; train the posterior separately.
    """
    if risk.ndim != 2 or risk.shape[1] == 0:
        raise ValueError("risk must be nonempty [batch, locations]")
    if not torch.isfinite(risk).all() or (risk < 0).any():
        raise ValueError("risk must be finite and nonnegative")
    a = risk.float()
    a = a.clamp_min(1e-6)
    a = a / a.mean(1, keepdim=True)
    s = torch.as_tensor(snr_linear, dtype=a.dtype, device=a.device)
    if s.numel() == 1:
        s = s.reshape(1, 1).expand(a.shape[0], 1)
    else:
        s = s.reshape(a.shape[0], 1)
    if not torch.isfinite(s).all() or (s <= 0).any():
        raise ValueError("SNR must be positive and finite")
    lo = torch.zeros_like(s)
    hi = (a * s).max(1, keepdim=True)[0]
    for _ in range(iterations):
        lam = (lo + hi) / 2
        p = ((a * s / lam.clamp_min(1e-30)).sqrt() - 1).clamp_min(0) / s
        exceeds = p.mean(1, keepdim=True) > 1
        lo = torch.where(exceeds, lam, lo)
        hi = torch.where(exceeds, hi, lam)
    p = ((a * s / ((lo + hi) / 2).clamp_min(1e-30)).sqrt() - 1).clamp_min(0) / s
    return (p / p.mean(1, keepdim=True).clamp_min(1e-12)).to(risk.dtype)


def depth_moments(probability, depth_samples):
    """Exact moments for a categorical depth posterior [B,D,H,W]."""
    z = torch.as_tensor(depth_samples, dtype=probability.dtype,
                        device=probability.device).reshape(1, -1, 1, 1)
    if z.shape[1] != probability.shape[1]:
        raise ValueError("posterior and depth sample counts differ")
    mean = (probability * z).sum(1, keepdim=True)
    variance = (probability * (z - mean).square()).sum(1, keepdim=True)
    return mean, variance
