"""Join fully closed source quality, actual rates and detector AP; no new fitting."""
import csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
def main():
 q=json.loads((ROOT/'data/provenance/source-quality-main-001-local-verification.json').read_text())
 d=json.loads((ROOT/'data/provenance/source-cache-inference-main-001-local-verification.json').read_text())
 rate=json.loads((ROOT/'data/provenance/original-source-cache-main-001-local-verification.json').read_text())
 assert d['state']=='passed_all45228_transferred_native_inference_records_predictions_and_metrics' and d['frames']==45228 and d['artifacts_verified']==45303
 rows=[]
 for codec in ('jpeg','jpeg2000'):
  for target in (10,30,50):
   key=f'{codec}-cr{target}';source=rate['conditions'][key];quality=q['conditions'][key]
   assert source['pairs']==3769 and quality['global_g']['views']==7538
   ap={name:d['metrics_by_endpoint'][key+'-'+name]['primary_Car_IoU_0_7_R40']['Car_3d/moderate_R40'] for name in ('stereo_rcnn','liga')}
   rows.append(dict(condition=key,codec=codec,nominal_CR=target,actual_CR=source['pooled_raw_to_wire_ratio'],source_bytes_per_pair=source['full_wire_bytes']/3769,pooled_full_PSNR=quality['global_g']['pooled_psnr_dB'],Stereo_RCNN_Car3D_R40_Moderate=ap['stereo_rcnn'],LIGA_Car3D_R40_Moderate=ap['liga']))
 out=ROOT/'experiments/original-paper-scenarios/source-task-main-results-001.csv';assert not out.exists()
 with out.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
 for ax,(field,title,ylabel) in zip(axes,[('pooled_full_PSNR','RGB source quality','Pooled PSNR (dB)'),('Stereo_RCNN_Car3D_R40_Moderate','Stereo R-CNN','Car 3D AP R40 (%)'),('LIGA_Car3D_R40_Moderate','LIGA-Stereo','Car 3D AP R40 (%)')]):
  for codec,color,marker,label in [('jpeg','#1763a6','o','JPEG'),('jpeg2000','#c04c27','s','JPEG2000')]:
   points=sorted([r for r in rows if r['codec']==codec],key=lambda r:r['source_bytes_per_pair'])
   ax.plot([r['source_bytes_per_pair']/1000 for r in points],[r[field] for r in points],marker=marker,color=color,label=label)
  ax.set(title=title,xlabel='Actual source kB / stereo pair',ylabel=ylabel);ax.grid(alpha=.2)
 axes[0].legend(frameon=False);fig.suptitle('KITTI val3769: source-only coding, IoU 0.7 / Moderate',fontsize=12)
 for suffix in ('png','pdf'):fig.savefig(ROOT/'to_human'/('source-quality-task-main-001.'+suffix),dpi=180)
 print(json.dumps(rows))
if __name__=='__main__':main()
