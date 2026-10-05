"""Plot completed checkpoint observations only; no matched-method gain/CI claim."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
f5=json.loads((ROOT/'data/provenance/stereo-feature-native-seed17-001-closure.json').read_text())
f6=json.loads((ROOT/'data/provenance/stereo-native-task-seed17-002-closure.json').read_text())
assert f5['state']==f6['state']=='closed_all_audits_passed'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(11.5,4.9),gridspec_kw={'width_ratios':[1.15,1]})
colors=['#4267a5','#d27a2c'];x=np.arange(2);width=.32
for index,channel in enumerate(['identity','awgn']):
 values=[r['evaluations'][channel]['Car3D_AP_R40_percent']['Car_3d/moderate_R40'] for r in [f5,f6]]
 bars=axes[0].bar(x+(index-.5)*width,values,width,color=colors[index],label='Identity' if channel=='identity' else 'AWGN 10 dB')
 for bar,value in zip(bars,values):axes[0].text(bar.get_x()+bar.get_width()/2,value+.6,f'{value:.2f}',ha='center',fontsize=10)
axes[0].set_xticks(x,['F5b\nFeature warmup','F6b\n+ 3D task adaptation']);axes[0].set_ylim(0,31)
axes[0].set_ylabel('Car Moderate 3D AP_R40 (%)');axes[0].set_title('Fixed 62,400-symbol detection');axes[0].legend(frameon=False,loc='upper left');axes[0].grid(axis='y',alpha=.18);axes[0].set_axisbelow(True)
features=['left_stereo','right_stereo','appearance'];x=np.arange(3)
for index,(r,label,color) in enumerate([(f5,'F5b','#7894bf'),(f6,'F6b','#985338')]):
 values=[r['evaluations']['identity']['feature_summaries'][name]['pooled_over_frames_nmse'] for name in features]
 bars=axes[1].bar(x+(index-.5)*width,values,width,color=color,label=label)
 for bar,value in zip(bars,values):axes[1].text(bar.get_x()+bar.get_width()/2,value*1.11,f'{value:.3f}',ha='center',fontsize=9)
axes[1].set_xticks(x,['Left stereo','Right stereo','Appearance']);axes[1].set_yscale('log');axes[1].set_ylim(.09,3.8)
axes[1].set_ylabel('Reference-feature NMSE (log scale)');axes[1].set_title('Identity feature distortion');axes[1].legend(frameon=False,loc='upper left');axes[1].grid(axis='y',alpha=.18);axes[1].set_axisbelow(True)
fig.suptitle('Task adaptation and feature distortion in native stereo transport',fontsize=13,y=.98)
fig.text(.5,.035,'One seed; fixed 372-frame exploratory holdout with author-pretraining overlap.\nF6b adds 3,340 GT updates; these observations do not establish equal-exposure method superiority.',ha='center',fontsize=9,color='#444444')
fig.tight_layout(rect=[0,.13,1,.93]);out=ROOT/'to_human/plots';out.mkdir(exist_ok=True)
for extension in ['png','pdf']:fig.savefig(out/f'F5b-F6b-task-feature-2026-10-04.{extension}',dpi=180,bbox_inches='tight')
