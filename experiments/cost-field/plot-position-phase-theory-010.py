"""Render complete, admitted phase mechanism results; no detector data."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[2]
r = json.loads((root/"data/provenance/position-phase-theory-local-010/report.json").read_text())
assert r["state"] == "passed_all130_exact_phase_moments_and_direct_Gaussian_checks"
assert len(r["rows"]) == 130 and all(row["passed"] for row in r["rows"])
fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.8), sharey=True, constrained_layout=True)
colors = ["#176a91", "#e07a23", "#7b4b9e"]
for ax, kind in zip(axes, ["awgn", "rayleigh"]):
    for theta, label, color in zip([0., 1/3, 1/2], [r"$\theta=0$ (midpoint)", r"$|\theta|=\pi/3$", r"$|\theta|=\pi/2$ (boundary)"], colors):
        rows = [x for x in r["rows"] if x["channel"] == kind and abs(x["theta_over_pi"]-theta) < 1e-12]
        assert len(rows) == 13
        ax.plot([x["snr_db"] for x in rows], [x["exact_rmse_m"] for x in rows], color=color, label=label)
        ax.scatter([x["snr_db"] for x in rows], [x["mc_rmse_m"] for x in rows], facecolors="none", edgecolors=color, s=23)
    ax.set(title="AWGN" if kind == "awgn" else "Rayleigh, perfect CSI / ZF", xlabel="SNR (dB)", xticks=np.arange(6, 19, 2))
    ax.grid(alpha=.2)
axes[0].set_ylabel("Packet coordinate RMSE (m)")
axes[0].legend(fontsize=8, frameon=False)
axes[1].text(.04, .06, "Lines: exact clipped-phase moment\nCircles: direct Gaussian simulation\n1,048,576 draws per condition", transform=axes[1].transAxes, va="bottom", fontsize=8)
fig.suptitle("Unit-energy phase coordinate over a 57.6 m interval", fontsize=12)
for suffix in ["png", "pdf"]:
    fig.savefig(root/f"to_human/packet-position-phase-theory-010.{suffix}", dpi=220)
