"""Complete independent preserved-row/mask audit; native adds actual RGB reads."""
import argparse,hashlib,io,json,math,time
from pathlib import Path
import numpy as np
from PIL import Image

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['native','local'],required=True);p.add_argument('--source-dir',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--data-root',type=Path);p.add_argument('--native-audit',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
 started=time.time();root=a.root.resolve();directory=a.directory.resolve();proof=read(directory/'merge.json');dpath=root/'data/provenance/cost-field-full-dataset-manifest-004.json';d=read(dpath)
 assert sha(dpath)==proof['dataset_manifest_sha256']=='ecf1f0e41de5757bd438b2a73252dc999eed80fa9828270beff342ccf6ea1310'
 assert proof['state']=='merged_unchanged_complete_public_train3712_ROI_records_pending_independent_audit'
 assert sha(directory/'records.jsonl')==proof['records_sha256']
 original={}
 for subset,n in [('train',3340),('holdout',372)]:
  path=a.source_dir/f'cao2025-roi-{subset}-001.jsonl';lines=path.read_bytes().splitlines(keepends=True);assert len(lines)==n
  for line in lines:
   row=json.loads(line);assert row['frame_id'] not in original;original[row['frame_id']]=line
 for name,item in proof['inputs'].items():
  path=a.source_dir/name;assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
 lines=(directory/'records.jsonl').read_bytes().splitlines(keepends=True);rows=[json.loads(x) for x in lines];ids=[r['frame_id'] for r in rows]
 assert ids==d['splits']['train']['ids'] and len(ids)==len(set(ids))==3712 and not set(ids)&set(d['splits']['val']['ids'])
 assert set(original)==set(ids) and all(line==original[frame] for line,frame in zip(lines,ids))
 views=boxes=pixels=empty=sourcebytes=0;areas=[];rgb_ledger=hashlib.sha256()
 if a.mode=='native':assert a.data_root is not None and a.native_audit is None
 else:
  assert a.native_audit is not None and a.data_root is None;native=read(a.native_audit)
  assert native['state']=='passed_all3712_preserved_rows_7424_native_RGB_sources_and_masks' and native['records_sha256']==proof['records_sha256'] and native['merge_sha256']==sha(directory/'merge.json')
 for row in rows:
  frame=row['frame_id'];assert [v['camera'] for v in row['views']]==['image_2','image_3']
  assert row['views'][0]['shape']==row['views'][1]['shape']
  for v in row['views']:
   h,w=v['shape'];assert isinstance(h,int) and isinstance(w,int) and min(h,w)>192
   rel=f"data/kitti/training/{v['camera']}/{frame}.png";sensor=d['files'][rel];assert sensor['sha256']==v['image_sha256']
   assert v['letterbox_shape'][:2]==[1,3] and all(0<x<=640 and x%32==0 for x in v['letterbox_shape'][2:])
   mask=np.zeros((h,w),np.uint8);assert len(v['boxes'])<=300
   for b in v['boxes']:
    x1,y1,x2,y2=b['xyxy'];assert all(type(x) is int for x in b['xyxy']) and 0<=x1<x2<=w and 0<=y1<y2<=h
    assert b['class_id'] in (2,5,7) and math.isfinite(b['confidence']) and .25<=b['confidence']<=1
    mask[y1:y2,x1:x2]=1;boxes+=1
   assert int(mask.sum())==v['union_area_pixels'] and hashlib.sha256(mask.tobytes()).hexdigest()==v['mask_uint8_sha256']
   if a.mode=='native':
    blob=(a.data_root/'training'/v['camera']/(frame+'.png')).read_bytes();assert len(blob)==sensor['bytes'] and hashlib.sha256(blob).hexdigest()==sensor['sha256']
    with Image.open(io.BytesIO(blob)) as im:
     assert im.mode=='RGB' and [im.height,im.width]==[h,w];rgb=np.asarray(im);assert rgb.dtype==np.uint8 and rgb.shape==(h,w,3)
    rgb_ledger.update((frame+':'+v['camera']+':').encode()+rgb.tobytes());sourcebytes+=len(blob)
   views+=1;pixels+=h*w;empty+=int(not v['boxes']);areas.append(float(mask.mean()))
  if len(areas)%1000==0:print(json.dumps(dict(frames=len(areas)//2,mode=a.mode)),flush=True)
 result=dict(state='passed_all3712_preserved_rows_7424_native_RGB_sources_and_masks' if a.mode=='native' else 'passed_all3712_transferred_preserved_rows_7424_mask_and_native_source_metadata',mode=a.mode,frames=3712,views=views,boxes=boxes,mask_pixel_values=pixels,empty_views=empty,mean_union_fraction=float(np.mean(areas)),records_sha256=sha(directory/'records.jsonl'),ordered_record_ledger_sha256=proof['records_sha256'],merge_sha256=sha(directory/'merge.json'),dataset_manifest_sha256=sha(dpath),verifier_sha256=sha(__file__),started_unix=started,ended_unix=time.time(),actual_native_RGB_bytes=sourcebytes if a.mode=='native' else None,native_RGB_pixel_ledger_sha256=rgb_ledger.hexdigest() if a.mode=='native' else native['native_RGB_pixel_ledger_sha256'],native_audit_sha256=sha(a.native_audit) if a.mode=='local' else None,GT_calibration_LiDAR_validation_RGB_read=False,limitation='All unchanged saved YOLO detections and union masks; no fresh YOLO inference, baseline training, radio or AP. Local scope uses native RGB proof and frozen full-data identities rather than reopening PNGs.')
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
