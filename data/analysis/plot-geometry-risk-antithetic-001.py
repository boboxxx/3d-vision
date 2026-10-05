"""Plot independently verified fresh8 frozen-predictor diagnostics."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2];PREFIX='geometry-risk-antithetic-001'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    path=ROOT/'data/analysis'/f'{PREFIX}-frozen-predictors-001.json';r=json.loads(path.read_text())
    ap=ROOT/'data/provenance'/f'{PREFIX}-statistics-local-audit-001.json';a=json.loads(ap.read_text())
    assert a['state']=='passed_independent_fresh8_frozen_predictor_statistics' and a['result_sha256']==sha(path)
    names=['energy','geometry','energy_geometry','gradient','shuffled_geometry']
    labels=['Code energy','Geometry','Energy + geometry','Gradient (labels required)','Shuffled geometry']
    colors=['#6b7280','#2563eb','#0891b2','#7c3aed','#d97706']
    keys=['signed_symmetric_Spearman','positive_symmetric_Spearman','antisymmetric_variance_Spearman']
    titles=['Signed mean increment','Positive part of mean increment','Antisymmetric fluctuation variance']
    fig,axes=plt.subplots(1,3,figsize=(14.5,4.8),sharey=True,layout='constrained')
    for ax,key,title in zip(axes,keys,titles):
        for i,(name,color) in enumerate(zip(names,colors)):
            s=r['models'][name]['summary'][key];m=s['mean'];ci=s['percentile_95_interval']
            if m is None:
                ax.text(0,i,'undefined',fontsize=9);continue
            ax.hlines(i,*ci,color=color,lw=2);ax.plot(m,i,'o',color=color,ms=7)
            ax.annotate(f'{m:.3f} ({s["defined_frames"]}/8)',(m,i),xytext=(0,11),textcoords='offset points',ha='center',fontsize=9)
        ax.set(title=title,yticks=np.arange(5),yticklabels=labels,ylim=(4.6,-.7),xlim=(-1,1),xlabel='Mean within-scene Spearman')
        ax.axvline(0,color='black',ls=':',lw=1);ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Frozen predictors on 8 fresh training scenes; 16 noise pairs per region',fontsize=14)
    fig.text(.5,-.055,'Bars: percentile 95% scene-bootstrap intervals. Exploratory, small sample; no allocation or AP result.',ha='center',fontsize=10)
    out=ROOT/'to_human/plots'/f'{PREFIX}-2026-10-04'
    for ext in ('png','svg'):fig.savefig(out.with_suffix('.'+ext),dpi=180,bbox_inches='tight')
    out.with_suffix('.json').write_text(json.dumps(dict(result_sha256=sha(path),local_audit_sha256=sha(ap),plotter_sha256=sha(Path(__file__)),model_order=names,metric_order=keys,bootstrap='1000 prelocked eight-scene resamples'),indent=2)+'\n')
    plt.close(fig)
if __name__=='__main__':main()
