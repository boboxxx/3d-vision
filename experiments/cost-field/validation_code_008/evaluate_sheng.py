"""Official author AP on all fixed transferred3769 native predictions."""
import argparse, hashlib, json, os, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--endpoint',type=Path,required=True);p.add_argument('--terminal',type=Path,required=True);p.add_argument('--dataset-root',type=Path,default=Path('/mnt/d/paper6/data/kitti'));p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
 reportpath=a.endpoint/'report.json';rowspath=a.endpoint/'frames.jsonl';r=json.loads(reportpath.read_text());terminal=json.loads(a.terminal.read_text())
 result=dict(state='starting',started_unix=time.time(),report_sha256=sha(reportpath),inference_terminal_sha256=sha(a.terminal),endpoint=str(a.endpoint),evaluator_sha256=sha(__file__),AP_endpoint_closed=False)
 def save():(a.output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 save()
 try:
  assert r['state']=='finished_full3769_final_epoch_inference_pending_terminal_wire_text_AP_closure' and r['frames_complete']==3769 and not r['AP_computed'] and not r['labels_GT_LiDAR_at_inference']
  assert terminal['actual_terminal'] and terminal['job_id']==r['job_id'] and terminal['report_sha256']==sha(reportpath)
  job=str(r['job_id']);lines=[x.split('|') for x in terminal['sacct'].splitlines()]
  assert any(x[0]==job and x[2:4]==['COMPLETED','0:0'] for x in lines)
  assert any(x[0]==job+'.0' and x[2:4]==['COMPLETED','0:0'] for x in lines)
  assert sha(rowspath)==r['frames_jsonl_sha256']
  assert r['author_initial_states']==r['author_final_states'] and r['codec_initial_states']==r['codec_final_states']
  assert r['sources_before']==r['sources_after'] and r['inference_sources_before']==r['inference_sources_after'] and r['base_freeze_before']==r['base_freeze_after']
  manifestpath=ROOT/'data/provenance/cost-field-full-dataset-manifest-004.json';assert sha(manifestpath)==r['manifest_sha256']
  manifest=json.loads(manifestpath.read_text());ids=manifest['splits']['val']['ids'];assert ids==r['validation_ids'] and len(ids)==len(set(ids))==3769
  assert {x.stem for x in (a.endpoint/'data').glob('*.txt')}==set(ids)
  evaluator_root='third_party/LIGA-Stereo/liga/datasets/kitti/kitti_object_eval_python/'
  evaluator_sources={p:h for p,h in r['sources_before'].items() if p.startswith(evaluator_root)}
  assert len(evaluator_sources)>=4
  for rel,h in evaluator_sources.items():assert sha(ROOT/rel)==h,rel
  authorhash=hashlib.sha256(json.dumps(r['author_initial_states'],sort_keys=True).encode()).hexdigest()
  codechash=hashlib.sha256(json.dumps(r['codec_initial_states'],sort_keys=True).encode()).hexdigest()
  prediction_hashes={};source_hashes={};total_boxes=0
  with rowspath.open() as f:
   for index,frame in enumerate(ids):
    row=json.loads(next(f));assert (row['index'],row['frame_id'],row['arm'],row['seed'],row['channel'],row['snr_db'])==(index,frame,r['arm'],r['seed'],r['channel'],r['snr_db'])
    assert row['original_states_sha256']==authorhash and row['codec_states_sha256']==codechash
    assert row['calls']==dict(detector=1,image_backbone=2,feature_neck=2,build_cost=1,head3D=1,depth_logits=1,forbidden=0,cut=1)
    assert set(row['inference_keys'])=={'batch_size','left_img','right_img','calib','image_shape','frame_id'}
    assert row['received_only_decode_identical_after_private_clear']
    for folder,ext in [('image_2','.png'),('image_3','.png'),('calib','.txt')]:
     rel=f'data/kitti/training/{folder}/{frame}{ext}';item=manifest['files'][rel];path=a.dataset_root/'training'/folder/(frame+ext)
     assert item==row['source'][folder] and path.stat().st_size==item['bytes'] and sha(path)==item['sha256'];source_hashes[rel]=item['sha256']
    pred=a.endpoint/'data'/(frame+'.txt');prediction_hashes[frame]=sha(pred);assert prediction_hashes[frame]==row['prediction_sha256']
    text=pred.read_text().splitlines();assert len(text)==row['prediction_count']
    assert all(len(line.split())==16 and line.split()[0] in ('Car','Pedestrian','Cyclist') and np.isfinite([float(v) for v in line.split()[1:]]).all() for line in text)
    total_boxes+=len(text)
   assert not f.read(1)
  # Full inference predictions are fixed and validated before any GT label read.
  label_hashes={}
  for frame in ids:
   rel=f'data/kitti/training/label_2/{frame}.txt';item=manifest['files'][rel];label=a.dataset_root/'training/label_2'/(frame+'.txt')
   assert label.stat().st_size==item['bytes'] and sha(label)==item['sha256'];label_hashes[frame]=item['sha256']
  result.update(state='running_official_full_validation_AP',prediction_hashes=prediction_hashes,source_hashes=source_hashes,label_hashes=label_hashes,evaluator_sources=evaluator_sources,total_boxes=total_boxes,frames=3769);save()
  usage=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()
  assert len(usage)==1 and int(usage[0])>=2048
  sys.path.insert(0,str(ROOT/'third_party/LIGA-Stereo/liga/datasets/kitti'))
  from kitti_object_eval_python import kitti_common
  from kitti_object_eval_python.eval import get_official_eval_result,do_eval
  gt=kitti_common.get_label_annos(a.dataset_root/'training/label_2',[int(v) for v in ids]);dt=kitti_common.get_label_annos(a.endpoint/'data',[int(v) for v in ids])
  text,strict=get_official_eval_result(gt,dt,['Car']);arrays=do_eval(gt,dt,[0],np.full((1,3,1),.5),compute_aos=True);relaxed={}
  for name,index in (('bbox',0),('bev',1),('3d',2),('aos',3)):
   for recall,offset in (('R11',0),('R40',4)):
    if arrays[index+offset] is not None:relaxed[name+'_'+recall]=arrays[index+offset][0,:,0].tolist()
  metrics=dict(primary_Car_IoU_0_7_R40={k:float(v) for k,v in strict.items()},secondary_Car_IoU_0_5=relaxed,difficulty_order=['easy','moderate','hard'])
  values=list(strict.values())+[v for row in relaxed.values() for v in row];assert np.isfinite(values).all() and all(0<=v<=100.000000001 for v in values)
  (a.output/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n');(a.output/'evaluator.txt').write_text(text)
  assert sha(reportpath)==result['report_sha256'] and sha(rowspath)==r['frames_jsonl_sha256']
  for frame,h in prediction_hashes.items():assert sha(a.endpoint/'data'/(frame+'.txt'))==h
  for frame,h in label_hashes.items():assert sha(a.dataset_root/'training/label_2'/(frame+'.txt'))==h
  for rel,h in evaluator_sources.items():assert sha(ROOT/rel)==h
  result.update(state='finished_official_full3769_AP_pending_actual_terminal_and_wire_local_closure',metrics=metrics,metrics_sha256=sha(a.output/'metrics.json'),evaluator_text_sha256=sha(a.output/'evaluator.txt'))
 except BaseException:result['state']='failed';result['traceback']=traceback.format_exc();raise
 finally:result['ended_unix']=time.time();save()
 print(json.dumps(dict(state=result['state'],frames=result['frames'],primary=result['metrics']['primary_Car_IoU_0_7_R40'])))
if __name__=='__main__':main()
