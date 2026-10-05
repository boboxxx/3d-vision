import os
os.environ['MPLCONFIGDIR']='/tmp/paper6-matplotlib-cache'
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/Users/chen/Documents/ChatGPT/paper6')
source=root/'data/provenance/pooling-diagnosis-seed17-002-closure.json';data=json.loads(source.read_text())
conditions=data['conditions'];values=[data['results'][c]['Car_3d/moderate_R40'] for c in conditions]
labels=['Unchanged control','Cost pooling only','Appearance pooling only','Cost + appearance pooling','Depth-preserved cost pooling']
fig,ax=plt.subplots(figsize=(8.4,4.6));fig.subplots_adjust(left=.32,right=.96,top=.79,bottom=.20)
ax.barh(range(5),values,color=['#355c7d','#b84a48','#51866b','#b84a48','#b84a48'],height=.62)
ax.set_yticks(range(5),labels);ax.invert_yaxis();ax.set_xlim(0,47)
ax.set_xlabel('Car Moderate 3D AP_R40 (%) at IoU 0.7')
for i,v in enumerate(values): ax.text(v+.55,i,f'{v:.4f}',va='center',fontsize=10)
ax.spines[['top','right']].set_visible(False);ax.xaxis.grid(True,alpha=.18);ax.set_axisbelow(True)
fig.suptitle('Raw-cost pooling damages the frozen detector',fontsize=14,y=.96)
fig.text(.32,.84,'Same F4 weights; 372 frames per condition; no codec/channel/training',fontsize=10)
fig.text(.02,.035,'Exploratory training-only holdout; author detector pretraining includes these frames.\nAll five fixed conditions audited. No communication-rate comparison or method-gain claim.',fontsize=9)
out=root/'to_human/plots';out.mkdir(exist_ok=True,parents=True)
fig.savefig(out/'pooling-diagnosis-2026-10-04.png',dpi=180);fig.savefig(out/'pooling-diagnosis-2026-10-04.pdf');plt.close(fig)
