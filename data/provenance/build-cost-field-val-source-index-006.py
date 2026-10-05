"""Public original sensor shape lineage from all six fully closed source caches."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 closurepath=ROOT/'data/provenance/original-source-cache-main-001-closure.json';proofpath=ROOT/'data/provenance/original-source-cache-main-001-local-verification.json'
 closure=json.loads(closurepath.read_text());proof=json.loads(proofpath.read_text())
 assert closure['state']=='closed_actual_terminal_all_native_artifacts' and closure['actual_terminal']
 assert proof['state']=='passed_all22614_transferred_pair_records_and_native_audit_metadata' and proof['closure_sha256']==sha(closurepath)
 encodepath=ROOT/'data/engineering/original-source-cache-main-001-transfer/mnt/d/paper6/runs/original-source-cache-main-001/encode.json'
 assert sha(encodepath)==closure['native_files']['/mnt/d/paper6/runs/original-source-cache-main-001/encode.json']
 encode=json.loads(encodepath.read_text());assert encode['state']=='finished'
 manifestpath=ROOT/'data/provenance/cost-field-full-dataset-manifest-004.json';manifest=json.loads(manifestpath.read_text());ids=manifest['splits']['val']['ids']
 assert encode['frame_ids']==ids and len(ids)==3769
 shapes={};checked=0
 for codec in ('jpeg','jpeg2000'):
  for rate in (10,30,50):
   rows=encode['conditions'][f'{codec}-cr{rate}']['records'];assert [r['frame_id'] for r in rows]==ids
   for r in rows:
    frame=r['frame_id'];views={};shape=None
    for folder,key in [('image_2','left_input'),('image_3','right_input')]:
     evidence=r['evidence'][key];assert evidence['dtype']=='|u1' and evidence['shape'][2]==3
     current=evidence['shape'][:2]
     if shape is None:shape=current
     else:assert shape==current
     rel=f'data/kitti/training/{folder}/{frame}.png';h=r['source_images_sha256'][f'/mnt/d/paper6/data/kitti/training/{folder}/{frame}.png']
     assert h==manifest['files'][rel]['sha256'];views[folder]=h
    value=dict(shape=shape,source_png_sha256=views)
    if frame in shapes:assert shapes[frame]==value
    else:shapes[frame]=value
    checked+=1
 assert set(shapes)==set(ids) and checked==22614
 out=ROOT/'data/provenance/cost-field-val-source-index-006.json';assert not out.exists()
 result=dict(state='closed_public_source_shapes_all3769_and_six_original_caches',ordered_ids=ids,frames=shapes,checked_cache_pairs=checked,manifest_sha256=sha(manifestpath),encode_sha256=sha(encodepath),original_closure_sha256=sha(closurepath),original_local_proof_sha256=sha(proofpath),builder_sha256=sha(__file__))
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(state=result['state'],frames=len(shapes),checked_cache_pairs=checked)))
if __name__=='__main__':main()
