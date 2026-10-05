"""Render sealed auxiliary calibration results, including all fixed controls."""
from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2];PREFIX='geometry-risk-pilot-001'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=ROOT/'data/analysis'/f'{PREFIX}-ridge-001.json';r=json.loads(p.read_text())
    a=ROOT/'data/provenance'/f'{PREFIX}-ridge-local-audit-001.json';audit=json.loads(a.read_text())
    assert audit['state']=='passed_independent_fixed_ridge_and_frame_bootstrap_audit' and audit['result_sha256']==sha(p)
    if not r['models']:raise ValueError('No common support; plot must not invent models')
    names=['energy','geometry','energy_geometry','gradient','shuffled_geometry']
    labels=['Code energy','Geometry','Energy + geometry','Gradient (diagnostic)','Shuffled geometry']
    colors=['#6b7280','#2563eb','#0891b2','#7c3aed','#d97706']
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for ax,key,title in zip(axes,['positive_Spearman','MSE'],['Positive damage: within-frame rank correlation','Positive damage: prediction error']):
        for y,(name,color) in enumerate(zip(names,colors)):
            s=r['models'][name]['summary'][key];m=s['mean'];ci=s['percentile_95_interval']
            if m is not None:
                ax.hlines(y,ci[0],ci[1],color=color,lw=2)
                ax.plot(m,y,'o',color=color,markersize=7)
                ax.annotate(f'{m:.3g} ({s["defined_frames"]}/32)',(m,y),xytext=(0,11),textcoords='offset points',ha='center',fontsize=9)
        ax.set(yticks=np.arange(5),yticklabels=labels,title=title,ylim=(4.6,-.7))
        ax.grid(axis='x',alpha=.25);ax.spines[['top','right']].set_visible(False)
    axes[0].axvline(0,color='black',ls=':',lw=1);axes[0].set_xlabel('Mean Spearman; percentile 95% frame bootstrap interval')
    axes[1].set_xlabel('Mean frame MSE; percentile 95% frame bootstrap interval')
    fig.suptitle('Fixed 32-frame fit / 32-frame auxiliary check',fontsize=14)
    fig.text(.5,-.055,'Training-set diagnostic, 4 masked-noise draws/group. Gradient uses labels. No AP or allocation result.',ha='center',fontsize=10)
    out=ROOT/'to_human/plots'/f'{PREFIX}-2026-10-04';out.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','svg'):fig.savefig(out.with_suffix('.'+suffix),dpi=180,bbox_inches='tight')
    out.with_suffix('.json').write_text(json.dumps(dict(result_sha256=sha(p),local_audit_sha256=sha(a),plotter_sha256=sha(Path(__file__)),models=names,metric_order=['positive_Spearman','MSE'],interval='1000 prelocked frame bootstrap resamples'),indent=2)+'\n')
    plt.close(fig)
if __name__=='__main__':main()
