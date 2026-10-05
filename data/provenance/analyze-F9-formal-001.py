"""Full closed F9 table/plot and explicitly exploratory recorded-gain summary."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
PREFIX='stereo-epipolar-native-seed17-001'
PROOF=ROOT/'data/provenance'/(PREFIX+'-local-verification-001.json')
OUTPUT=ROOT/'data/provenance'/(PREFIX+'-analysis-001.json')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert not OUTPUT.exists();proof=json.loads(PROOF.read_text())
    closure=ROOT/'data/provenance'/(PREFIX+'-closure.json')
    assert sha(closure)==proof['closure_sha256'] and proof['state']=='passed_all_transferred_artifacts13360_training_rows16_final_endpoints'
    conditions=('identity10','awgn6','awgn10','awgn18');arms=('U','G','P','S')
    labels=dict(U='Uniform',G='Generic head',P='Epipolar coupling',S='Shuffled coupling')
    table=[];gain_summary={}
    for arm in arms:
        for condition in conditions:
            ap=proof['metrics'][arm+'-'+condition]
            table.append(dict(arm=arm,condition=condition,**{difficulty:ap['Car_3d/'+difficulty+'_R40'] for difficulty in ('easy','moderate','hard')}))
        path=ROOT/'data/runs'/(PREFIX+'-'+arm+'-training-audit.records')/'training.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines()];assert len(rows)==3340
        # Post-result descriptive window; no intervention, AP selection or causal claim.
        gains=np.array([r['F9_coding']['amplitude_gains'] for r in rows[-500:]])
        gain_summary[arm]=dict(raw_records_sha256=sha(path),last_steps=[2841,3340],
                              mean_absolute_gain_deviation=float(np.abs(gains-1).mean()),
                              median_per_frame_gain_span=float(np.median(np.ptp(gains,axis=1))),
                              minimum=float(gains.min()),maximum=float(gains.max()))
    result=dict(state='closed_result_analysis',local_proof_sha256=sha(PROOF),table=table,
                prelocked_primary_differences_pp=proof['prelocked_AWGN10_Moderate_differences_pp'],
                geometric_primary_supported=False,
                exploratory_post_AP_gain_summary=gain_summary,
                limitation='Single internal372/seed17, author-pretraining exposure; no main/multiseed/statistical superiority or overall geometry rejection')
    with OUTPUT.open('x') as stream:json.dump(result,stream,indent=2)
    directory=ROOT/'experiments/geometry-link/F9';csv_path=directory/'formal-results-001.csv'
    with csv_path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['arm','condition','easy','moderate','hard']);writer.writeheader();writer.writerows(table)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),gridspec_kw={'width_ratios':[1.5,1]})
    colors=dict(U='#64748b',G='#2563eb',P='#d97706',S='#9333ea')
    for arm in arms:
        y=[proof['metrics'][arm+'-'+condition]['Car_3d/moderate_R40'] for condition in conditions]
        axes[0].plot(range(4),y,marker='o',color=colors[arm],label=labels[arm])
    axes[0].set_xticks(range(4),['Identity','AWGN 6','AWGN 10','AWGN 18'])
    axes[0].set_ylabel('Car Moderate 3D AP_R40 (%)');axes[0].grid(axis='y',alpha=.2);axes[0].legend(fontsize=8)
    differences=proof['prelocked_AWGN10_Moderate_differences_pp'];values=[differences['P_minus_'+a] for a in ('G','S','U')]
    axes[1].barh([2,1,0],values,color=['#b91c1c' if x<0 else '#047857' for x in values])
    axes[1].set_yticks([2,1,0],['P − generic','P − shuffled','P − uniform']);axes[1].axvline(0,color='#475569',lw=.8)
    axes[1].set_xlim(-.65,1.15);axes[1].set_xlabel('AWGN10 AP difference (percentage points)')
    for y,x in zip((2,1,0),values):axes[1].text(x+(.035 if x>=0 else -.035),y,f'{x:+.3f}',ha='left' if x>=0 else 'right',va='center',fontsize=9)
    fig.suptitle('F9: prelocked geometry contrast is not supported',fontsize=12)
    fig.text(.5,.015,'Internal 372 frames • seed17 • IoU0.7 • equal 62,400 complex uses • no confidence interval',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.05,1,.94]);plots=ROOT/'to_human/plots';plots.mkdir(exist_ok=True)
    for suffix in ('png','pdf'):
        target=plots/('F9-epipolar-2026-10-04.'+suffix);assert not target.exists();fig.savefig(target,dpi=180)
    plt.close(fig);print(json.dumps(result['prelocked_primary_differences_pp']))


if __name__=='__main__':main()
