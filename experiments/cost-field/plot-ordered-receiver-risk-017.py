"""Exploratory visualization of closed physical-prior evidence, no AP claim."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
report = json.loads((ROOT/'data/engineering/ordered-receiver-local-016.json').read_text())
closure = json.loads((ROOT/'data/provenance/ordered-receiver-two-host-closure-017.json').read_text())
assert closure['state'] == 'closed_actual_local_sheng_full016_and_all_retained_paired_errors017'
conditions = report['declared_prior_risk']['conditions']
styles = {
    'hard': ('Clipped phase', '#64748b', '--'),
    'independent': ('Independent posterior', '#2563eb', '-.'),
    'isotonic': ('Weighted isotonic', '#0891b2', ':'),
    'ordered_blind': ('Ordered, fading-blind', '#d97706', '--'),
    'ordered': ('Ordered + actual CSI', '#7e22ce', '-'),
    'ordered_mean': ('Basis at posterior mean', '#c084fc', ':'),
}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0), sharex='col')
handles = {}
for col, channel in enumerate(['awgn', 'rayleigh']):
    cases = [c for c in conditions if c['channel'] == channel]
    x = [c['SNR_dB'] for c in cases]
    axes[0, col].set_title('AWGN' if channel == 'awgn' else 'Iid Rayleigh, perfect receiver CSI')
    for name, (label, color, linestyle) in styles.items():
        if name == 'ordered_blind' and channel == 'awgn':
            continue
        if name != 'ordered_mean':
            y = np.sqrt([c['coordinate_latent_m2_MSE'][name] for c in cases])
            handles[name], = axes[0, col].plot(x, y, color=color, linestyle=linestyle,
                marker='o' if name == 'ordered' else None, markersize=3, linewidth=1.6, label=label)
        y = [c['fixed_coefficient_field_MSE'][name]['mean'] for c in cases]
        line, = axes[1, col].plot(x, y, color=color, linestyle=linestyle,
            marker='o' if name == 'ordered' else None, markersize=3, linewidth=1.6, label=label)
        handles.setdefault(name, line)
    for row in (0, 1):
        axes[row, col].grid(alpha=.2)
        axes[row, col].set_xlim(6, 18)
        axes[row, col].set_xticks([6, 8, 10, 12, 14, 16, 18])
    axes[1, col].set_yscale('log')
    axes[1, col].set_xlabel('Mean SNR (dB)')
axes[0, 0].set_ylabel('Latent node RMSE (m)')
axes[1, 0].set_ylabel('Fixed-coefficient basis-field MSE')
fig.legend([handles[k] for k in styles], [styles[k][0] for k in styles],
    loc='lower center', bbox_to_anchor=(.5, .035), ncol=3, frameon=False, fontsize=8)
fig.text(.5, .005, 'Declared 33-point ordered-grid prior; 65,536 trials per condition. Synthetic latent risk, not KITTI object depth or AP.',
    ha='center', fontsize=7)
fig.tight_layout(rect=(0, .14, 1, 1))
for suffix in ['png', 'pdf']:
    fig.savefig(ROOT/'to_human'/f'ordered-receiver-risk-017.{suffix}', dpi=300)
plt.close(fig)
