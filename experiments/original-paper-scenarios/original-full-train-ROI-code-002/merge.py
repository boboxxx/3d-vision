"""Reuse unchanged sensor-only ROI rows in the complete public train order."""
import argparse,hashlib,json
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())

def main():
 p=argparse.ArgumentParser();p.add_argument('--source-dir',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();src=a.source_dir.resolve();out=a.output_dir.resolve();assert not out.exists()
 dataset=root/'data/provenance/cost-field-full-dataset-manifest-004.json';fold=root/'data/internal-tuning-fold-001.json';meta=root/'reproduction/cao2025/upstream/yolov5.json'
 assert sha(dataset)=='ecf1f0e41de5757bd438b2a73252dc999eed80fa9828270beff342ccf6ea1310'
 d=read(dataset);f=read(fold);m=read(meta);ids=d['splits']['train']['ids'];val=d['splits']['val']['ids']
 assert len(ids)==len(set(ids))==3712 and len(val)==3769 and not set(ids)&set(val)
 inputs={};rows={};origin={}
 for name,split,count,expected in [('train','geocomm_tune_train',3340,'f7413abe7a587ac2c09ba5a3e5a4851f20db8bdfb8b7a3a5e1c90ac0ee87fad2'),('holdout','geocomm_tune_holdout',372,'883a35bbe3e7a3dbaccc75e71406512360a1d844f65400239a53cfb5ed7dadde')]:
  rec=src/f'cao2025-roi-{name}-001.jsonl';manifest=src/f'cao2025-roi-{name}-001.manifest.json';audit=src/f'cao2025-roi-{name}-audit-001.json';x=read(manifest);y=read(audit)
  assert sha(rec)==expected==x['records_sha256']==y['records_sha256']
  assert y['state']=='passed' and y['manifest_sha256']==sha(manifest)
  assert x['state']=='finished' and x['evidence_type']=='sensor_only_YOLO_ROI_fold_extraction' and x['frames']==y['frames']==count
  assert x['split']==y['split']==split and x['fold_sha256']==y['fold_sha256']==sha(fold)
  assert x['checkpoint_sha256']==m['checkpoint_sha256'] and x['upstream_revision']==m['revision'] and x['metadata_sha256']==sha(meta)
  assert x['source_sha256']==sha(root/'reproduction/cao2025/extract_rois.py')
  lines=rec.read_bytes().splitlines(keepends=True);parsed=[json.loads(line) for line in lines]
  assert [r['frame_id'] for r in parsed]==f['folds'][split]['ids']
  for line,r in zip(lines,parsed):
   frame=r['frame_id'];assert frame not in rows;rows[frame]=line;origin[frame]=name
  for q in [rec,manifest,audit]:inputs[q.name]=dict(bytes=q.stat().st_size,sha256=sha(q))
 assert set(rows)==set(ids) and not set(rows)&set(val)
 out.mkdir(parents=True);records=out/'records.jsonl';records.write_bytes(b''.join(rows[frame] for frame in ids))
 proof=dict(state='merged_unchanged_complete_public_train3712_ROI_records_pending_independent_audit',frames=3712,views=7424,records_sha256=sha(records),inputs=inputs,dataset_manifest_sha256=sha(dataset),fold_sha256=sha(fold),YOLO_metadata_sha256=sha(meta),builder_sha256=sha(__file__),protocol_sha256=sha(root/'experiments/original-paper-scenarios/original-full-train-ROI-protocol-002.md'),source_subsets={'train':3340,'holdout':372},public_validation_frames_used=0,GT_calibration_LiDAR_read=False)
 (out/'merge.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
if __name__=='__main__':main()
