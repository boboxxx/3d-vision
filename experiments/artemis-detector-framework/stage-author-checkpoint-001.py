"""Stage only the public author LIGA checkpoint, verify before acceptance."""
import hashlib
import json
from pathlib import Path
import time
import traceback
import urllib.request
ROOT=Path(__file__).resolve().parents[2]
DIR=ROOT/'assets/liga-native-001';OUT=ROOT/'data/engineering/artemis-liga-author-checkpoint-001.json'
URL='https://drive.usercontent.google.com/download?id=1M97Wa2ehAdCD6rwEhe67Vde7y4fLtVXo&export=download&confirm=t'
EXPECTED='3f36af30b686f23d0868482675177b29cc50c2b378577dd12f9092ddb0235b2e'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not OUT.exists();DIR.mkdir(parents=True,exist_ok=False)
 paths=['experiments/artemis-detector-framework/protocol-009.md','experiments/artemis-detector-framework/stage-author-checkpoint-001.py','third_party/LIGA-Stereo/README.md']
 source={p:sha(ROOT/p) for p in paths}
 r=dict(state='running',url=URL,started_unix=time.time(),sources=source,expected_sha256=EXPECTED,expected_bytes=93383301,deserialized=False,GPU_used=False)
 def save():OUT.write_text(json.dumps(r,indent=2)+'\n')
 save()
 try:
  req=urllib.request.Request(URL,headers={'User-Agent':'Mozilla/5.0'})
  with urllib.request.urlopen(req,timeout=45) as response,(DIR/'author-checkpoint.download').open('xb') as f:
   r['http_status']=response.status;r['resolved_url']=response.url;r['content_type']=response.headers.get('Content-Type');save()
   h=hashlib.sha256();size=0
   while chunk:=response.read(1024*1024):
    f.write(chunk);h.update(chunk);size+=len(chunk)
    assert size<=93383301,'payload larger than locked checkpoint'
  r['actual_bytes']=size;r['actual_sha256']=h.hexdigest();save()
  assert size==93383301 and h.hexdigest()==EXPECTED,'keep unaccepted payload; full digest/size mismatch'
  (DIR/'author-checkpoint.download').rename(DIR/'liga-author-download')
  r['accepted_path']='assets/liga-native-001/liga-author-download';r['state']='passed_complete_author_checkpoint_bytes_no_deserialization_or_GPU'
 except BaseException:
  r['state']='failed';r['traceback']=traceback.format_exc();raise
 finally:
  r['sources_after']={p:sha(ROOT/p) for p in paths};assert r['sources_after']==source
  r['ended_unix']=time.time();save()
if __name__=='__main__':main()
