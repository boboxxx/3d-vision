"""Immutable whole raw stereo dataset identity before full task training."""
import hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 out=ROOT/'data/provenance/cost-field-full-dataset-manifest-004.json';assert not out.exists()
 root=ROOT/'data/kitti';splits={};files={};start=time.time()
 for split,n,digest in [('train',3712,'b6417a1d9b18c8fdb085128e633d28ff321b7674a6d1b3841b8f43d865b281cb'),('val',3769,'657ac4bcc1e156e5b106a4ca18e1f88e012787ea1d2b5d0adeea97fee903fa86')]:
  p=root/'ImageSets'/(split+'.txt');assert sha(p)==digest
  ids=p.read_text().splitlines();assert len(ids)==len(set(ids))==n and all(len(x)==6 and x.isdecimal() for x in ids)
  splits[split]=dict(ids=ids,sha256=digest,path=str(p.relative_to(ROOT)))
 assert not set(splits['train']['ids'])&set(splits['val']['ids'])
 for frame in sorted(splits['train']['ids']+splits['val']['ids']):
  for folder,ext in [('image_2','png'),('image_3','png'),('calib','txt'),('label_2','txt')]:
   p=root/'training'/folder/(frame+'.'+ext);assert p.is_file();files[str(p.relative_to(ROOT))]=dict(bytes=p.stat().st_size,sha256=sha(p))
 r=dict(state='passed_whole_raw_KITTI_stereo_train3712_val3769_bytes_and_disjoint_splits',started_unix=start,ended_unix=time.time(),splits=splits,files=files,total_files=len(files),total_bytes=sum(x['bytes'] for x in files.values()),verifier_sha256=sha(Path(__file__)))
 assert len(files)==29924;out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ('state','total_files','total_bytes','started_unix','ended_unix')}))
if __name__=='__main__':main()
