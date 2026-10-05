"""All synthetic intrinsic distortions and the fixed equal-mass mode limitation."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
folder=ROOT/'data/engineering/depth-particles-Q0-local-CPU-001'
report=json.loads((folder/'report.json').read_text())
with np.load(folder/'complete_arrays.npz',allow_pickle=False) as archive:
    p=archive['probabilities'][6];edges=archive['depth_edges'];atoms=archive['particles'][6]
names=['Narrow near','Narrow middle','Narrow far','Uniform','Broad middle','Broad far',
       'Two modes 50/50','Two modes 35/65','Far mode 5%','Far mode 10%','Far mode 20%','Far mode 40%']
values=list(report['intrinsic_W1_m_by_source'].values())
fig,(ax,left)=plt.subplots(1,2,figsize=(12,5.8),gridspec_kw={'width_ratios':[1.1,1]})
ax.barh(names,values,color=['#cd6848' if i==6 else '#416f92' for i in range(12)])
ax.invert_yaxis();ax.set_xlabel('Intrinsic W1 error (m), noiseless channel')
ax.set_xlim(0,9.4);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
for i,value in enumerate(values):ax.text(value+.08,i,f'{value:.3f}',va='center',fontsize=8)
ax.set_title('All 12 fixed synthetic distributions',fontsize=11)
x=np.linspace(edges[0],edges[-1],4001);indices=np.clip(np.searchsorted(edges,x,side='right')-1,0,71)
prior=np.r_[0,p.cumsum()[:-1]];cdf=prior[indices]+p[indices]*(x-edges[indices])/(edges[indices+1]-edges[indices])
coded=(x[:,None]>=atoms[None,:]).mean(-1)
left.plot(x,cdf,label='Source two-mode histogram',color='#416f92',linewidth=2)
left.step(x,coded,where='post',label='Three equal-mass atoms',color='#cd6848',linewidth=1.8)
left.fill_between(x,cdf,coded,color='#cd6848',alpha=.16)
left.set(xlabel='Metric depth (m)',ylabel='CDF',ylim=(-.03,1.06),title='50/50 modes: 8 m error before transmission')
left.grid(alpha=.2);left.legend(fontsize=8,loc='lower right')
fig.suptitle('Q0 feasibility: source approximation loss remains at infinite SNR',fontsize=12)
fig.text(.5,.01,'Synthetic CPU experiment; no KITTI, detector or AP result. Two charged complex symbols per cell.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.035,1,.94))
output=ROOT/'to_human/depth-particles-Q0-intrinsic-001.png';assert not output.exists();fig.savefig(output,dpi=180)
print(str(output))
