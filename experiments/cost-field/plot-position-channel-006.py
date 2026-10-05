"""Standalone scientific plot of all sealed G256 packet-coordinate diagnostics."""
import json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
r=json.loads((ROOT/'data/provenance/cost-field-position-channel-006.json').read_text())
assert r['state']=='passed_whole_sealed_G256_prefix_coordinate_diagnostic' and r['position_nodes']==1198080 and r['checked_updates']==256
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
fig,ax=plt.subplots(1,2,figsize=(11,4.7));x=np.arange(6);labels=[f'[{6+2*i}, {8+2*i})' for i in range(6)]
colors={'awgn':'#2563eb','rayleigh':'#c2410c'}
for c in ('awgn','rayleigh'):
 b=sorted([v for v in r['bins'] if v['channel']==c],key=lambda v:v['SNR_lower_db']);assert len(b)==6
 ax[0].plot(x,[v['RMSE_coordinate_m'] for v in b],'o-',label=c.upper(),color=colors[c],lw=2)
 ax[1].plot(x,[v['clipping_fraction']*100 for v in b],'o-',label=c.upper(),color=colors[c],lw=2)
 for i,v in enumerate(b):ax[0].annotate(f"{v['RMSE_coordinate_m']:.2f}",(i,v['RMSE_coordinate_m']),xytext=(0,6),textcoords='offset points',ha='center',color=colors[c],fontsize=8)
low=np.arange(6,18,2);high=low+2
mean_N0=10/(math.log(10)*2)*(10.**(-low/10)-10.**(-high/10))
approx=57.6/math.pi*np.sqrt(mean_N0/2)
ax[0].plot(x,approx,'--',color='#64748b',label='AWGN interior linear approximation',lw=1.4)
ax[0].set_ylabel('Packet coordinate displacement RMS (m)');ax[1].set_ylabel('Decoder half-arc clipping (%)')
for a in ax:
 a.set_xticks(x,labels);a.set_xlabel('Observed SNR interval (dB)');a.grid(axis='y',alpha=.25);a.spines[['top','right']].set_visible(False)
ax[0].legend(fontsize=8,loc='upper right');ax[1].legend(fontsize=9)
fig.suptitle('Explicit position symbols incur channel displacement',fontsize=13)
fig.text(.5,.90,'G / seed 17 / first 256 sealed updates — packet coordinates, not object depth errors',ha='center',fontsize=9)
fig.text(.5,.025,'All 1,198,080 position symbols. Physical fading remains unclipped. No validation/AP result.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.055,1,.88))
for ext in ('png','pdf'):fig.savefig(ROOT/f'to_human/packet-position-channel-006.{ext}',dpi=180,bbox_inches='tight')
print('saved standalone PNG/PDF')
