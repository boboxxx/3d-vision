import math
import torch
from torch import nn


class ComplexChannel(nn.Module):
    """Unit complex-symbol energy AWGN or pilot-estimated frame Rayleigh.

    Input/output [B,N,2], where last dimension is real/imaginary. Receiver
    uses only noisy pilots and y. No clean scale, power map or exact h enters
    its output. Static layout/rate and camera calibration are preprovisioned.
    """
    def __init__(self, kind="awgn", pilots=8):
        super().__init__()
        if kind not in ("awgn", "rayleigh", "identity"):
            raise ValueError("unsupported channel " + kind)
        if kind == "rayleigh" and pilots < 1:
            raise ValueError("Rayleigh requires explicit pilots")
        self.kind, self.pilots = kind, pilots

    def forward(self, symbols, snr_db, generator=None):
        if symbols.ndim != 3 or symbols.shape[-1] != 2:
            raise ValueError("symbols must be [batch, complex uses, 2]")
        if not math.isfinite(float(snr_db)):
            raise ValueError("SNR must be finite")
        n0 = 10 ** (-float(snr_db) / 10)
        noise_std = math.sqrt(n0 / 2)
        def randn(shape):
            return torch.randn(shape, dtype=symbols.dtype, device=symbols.device,
                               generator=generator)
        pilot_count = self.pilots if self.kind == "rayleigh" else 0
        energy = symbols.square().sum((1, 2))
        accounting = {
            "data_complex_uses": symbols.shape[1],
            "pilot_complex_uses": pilot_count,
            "header_complex_uses": 0,
            "total_complex_uses": symbols.shape[1] + pilot_count,
            "tx_energy_per_frame": energy.detach() + pilot_count,
            "snr_db": float(snr_db), "channel": self.kind,
            "receiver_csi": "noisy_pilots" if pilot_count else "not_needed",
        }
        if self.kind == "identity":
            return symbols, accounting
        if self.kind == "awgn":
            return symbols + randn(symbols.shape) * noise_std, accounting
        h = randn((symbols.shape[0], 1, 2)) / math.sqrt(2)
        hr, hi = h[..., 0], h[..., 1]
        xr, xi = symbols[..., 0], symbols[..., 1]
        y = torch.stack((hr * xr - hi * xi, hr * xi + hi * xr), -1)
        y = y + randn(y.shape) * noise_std
        # Every pilot is 1+0j with energy 1. Estimate h from noisy observations.
        yp = h + randn((symbols.shape[0], self.pilots, 2)) * noise_std
        hhat = yp.mean(1, keepdim=True) / (1 + n0 / self.pilots)
        ar, ai = hhat[..., 0], hhat[..., 1]
        denom = ar.square() + ai.square() + n0
        r = torch.stack(((ar * y[..., 0] + ai * y[..., 1]) / denom,
                         (ar * y[..., 1] - ai * y[..., 0]) / denom), -1)
        return r, accounting
