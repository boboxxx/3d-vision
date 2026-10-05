"""Plot only the independently closed prelocked F8 endpoints."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
PREFIX='stereo-encoder-native-seed17-001'
closure=json.loads((ROOT/'data/provenance'/f'{PREFIX}-closure.json').read_text())
audit=json.loads((ROOT/'data/provenance'/f'{PREFIX}-local-verification-001.json').read_text())
assert closure['state']=='closed_all_audits_passed' and audit['state']=='passed'
values={arm:[closure['evaluations'][f'{PREFIX}-{arm}-test-{channel}']['Car3D_AP_R40_percent']['Car_3d/moderate_R40']
             for channel in ('identity','awgn')] for arm in ('codec','joint')}
fig,ax=plt.subplots(figsize=(8.4,5.6))
fig.subplots_adjust(bottom=.25,top=.82,left=.12,right=.97)
fig.suptitle('Matched encoder adaptation',x=.12,ha='left',fontsize=17,fontweight='bold')
ax.set_title('Internal 372-frame holdout · Car 3D IoU 0.7',loc='left',fontsize=11,pad=14)
x=np.arange(2);width=.32
for offset,arm,label,color in [(-width/2,'codec','Codec only (16 tensors)','#336a94'),(width/2,'joint','Encoder + codec (51 tensors)','#d87c3e')]:
    bars=ax.bar(x+offset,values[arm],width,label=label,color=color)
    ax.bar_label(bars,labels=[f'{v:.2f}' for v in values[arm]],padding=5,fontsize=11)
ax.set_xticks(x,['Identity channel','AWGN 10 dB'])
ax.set_ylabel('Moderate AP_R40 (%)')
ax.set_ylim(0,max(max(v) for v in values.values())*1.3)
ax.legend(frameon=False,loc='upper left',fontsize=10)
ax.spines[['top','right']].set_visible(False)
ax.yaxis.grid(True,color='#e5e5e5');ax.set_axisbelow(True)
delta=closure['primary_AWGN10_Moderate_treatment_minus_control_pp']
fig.text(.12,.125,f'Prelocked AWGN10 difference: {delta:+.4f} percentage points',fontsize=11,fontweight='bold')
fig.text(.12,.055,'Same parent · 3,340 updates/arm · paired data and noise · 62,400 complex uses/frame\nSingle seed; author-pretraining overlap. No main-validation or geometry-allocation claim.',fontsize=9,color='#444444')
directory=ROOT/'to_human/plots';directory.mkdir(exist_ok=True)
for extension in ('png','svg'):
    output=directory/f'F8-matched-encoder-2026-10-04.{extension}';assert not output.exists()
    fig.savefig(output,dpi=180,facecolor='white')
plt.close(fig)
print(json.dumps(dict(values=values,primary_delta_pp=delta)))
