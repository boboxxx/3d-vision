"""Publication-style summary of closed representation feasibility, not task error."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
n=json.loads((ROOT/'data/provenance/depth-particles-native-Q2-local-verification-002.json').read_text())
assert n['state']=='passed_all_native_Q2_cells_independent_spatial_CDF_and_direct_partition_replay'
s=json.loads((ROOT/'data/engineering/depth-particles-Q1-local-CPU-001/report.json').read_text())
awgn=next(c for c in s['conditions'] if c['channel']=='awgn' and c['SNR_dB']==10)
source_names=['bimodal_50_50','far_minority_5pct','trimodal_equal']
labels=['Two modes, 50/50','Far minority, 5%','Three modes']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(10,4.1),layout='constrained')
x=np.arange(3);colors=['#3569a8','#c27b30','#4e8468']
for i,method in enumerate(['Q0','Q1']):
 vals={r['name']:r['full_W1_mean_m'] for r in awgn['methods'][method]['rows']}
 axs[0].bar(x+(i-.5)*.34,[vals[k] for k in source_names],width=.34,color=colors[i],label=method)
axs[0].set_xticks(x,labels,rotation=15,ha='right');axs[0].set_ylabel('Distribution W1 (m)');axs[0].set_title('Synthetic laws, AWGN 10 dB');axs[0].legend(frameon=False)
methods=['Q0','Q1','Q2'];x=np.arange(2)
for i,method in enumerate(methods):
 vals=[n['pooled'][c+'_'+method]['mean_W1_m'] for c in ['identity','awgn10']]
 bars=axs[1].bar(x+(i-1)*.24,vals,width=.24,color=colors[i],label=method)
 axs[1].bar_label(bars,fmt='%.2f',fontsize=8,padding=3)
axs[1].set_xticks(x,['Identity','AWGN 10 dB']);axs[1].set_ylim(0,16);axs[1].set_title('98,199 cached sender laws');axs[1].set_ylabel('Mean distribution W1 (m)');axs[1].legend(frameon=False,ncol=3)
fig.suptitle('Better source approximation does not ensure better noisy transport',fontsize=12)
fig.savefig(ROOT/'to_human/depth-particles-Q1-Q2-002.png',dpi=180)
