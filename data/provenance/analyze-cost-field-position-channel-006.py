"""Whole explicit sealed G256 prefix coordinate-channel diagnostics."""
import argparse,hashlib,importlib.util,json,math,sys,tarfile,io
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'data/provenance/audit-cost-field-full-006.py';s=importlib.util.spec_from_file_location('audit006',BASE);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
SNAPSHOT=ROOT/'data/provenance/cost-field-full-report-snapshot-004-003-seed17.json'

def export():
 snap=json.loads((ROOT/'data/provenance/cost-field-full-actual-snapshot-004-003.json').read_text());b=SNAPSHOT.read_bytes();assert m.digest(b)==snap['runs']['17']['report_snapshot_sha256'];r=json.loads(b);assert r['seed']==17 and r['update_count']>=256
 directory=ROOT/'data/runs/cost-field-full-seed17-004'
 def blobs():
  yield 'metadata.json',json.dumps(dict(seed=17,count=256,whole=False,terminal=None,report_sha256=m.digest(b))).encode()
  yield 'report.json',b
  yield 'codec-initial.pt',(directory/'codec-initial.pt').read_bytes()
  seen=set();h=hashlib.sha256()
  with (directory/'updates.jsonl').open('rb') as f:
   for index in range(256):
    line=f.readline();h.update(line);row=json.loads(line);yield f'row/{index}.json',line;frame=row['frame_id']
    assert frame not in seen;seen.add(frame)
    for folder,ext in [('image_2','.png'),('image_3','.png'),('calib','.txt'),('label_2','.txt')]:
     rel=f'data/kitti/training/{folder}/{frame}{ext}';yield rel,(ROOT/rel).read_bytes()
    yield row['artifact'],(ROOT/row['artifact']).read_bytes();yield row['optimizer_checkpoint'],(ROOT/row['optimizer_checkpoint']).read_bytes()
  yield 'footer.json',json.dumps(dict(count=256,updates_jsonl_sha256=h.hexdigest(),unique_frames=256)).encode()
 with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
  for name,data in blobs():
   info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data))

class Positions(m.Audit):
 def __init__(self,*args):
  super().__init__(*args);self.bins={(c,i):dict(frames=0,nodes=0,abs_sum=0.,squared_sum=0.,signed_sum=0.,max_abs=0.,clipped=0,reversed_adjacent=0,adjacent_pairs=0) for c in ('awgn','rayleigh') for i in range(6)}
 def physics(self,a,row):
  super().physics(a,row);assert row['arm']=='G'
  tx=a['wire_tx'].astype(np.float64).reshape(-1,19,2)[:,:3];rx=a['wire_received'].astype(np.float64).reshape(-1,19,2)[:,:3]
  axis=self.initial['G']['depth_axis'].numpy().astype(np.float64);span=axis[-1]-axis[0]
  theta=np.arctan2(tx[...,1],tx[...,0]);assert (np.abs(theta)<=math.pi/2+64*m.EPS).all()
  angle=np.arctan2(rx[...,1],rx[...,0]);angle=np.where(np.max(np.abs(rx),axis=-1)<=m.EPS,0.,angle)
  clipped=np.clip(angle,-math.pi/2,math.pi/2)
  source=axis[0]+span/2+span/math.pi*theta;received=axis[0]+span/2+span/math.pi*clipped;delta=received-source
  assert np.isfinite(delta).all() and (np.abs(delta)<=span+64*m.EPS*span).all()
  index=int((row['snr_db']-6)//2);assert 0<=index<6;stats=self.bins[(row['channel'],index)]
  stats['frames']+=1;stats['nodes']+=delta.size;stats['abs_sum']+=float(np.abs(delta).sum());stats['squared_sum']+=float((delta*delta).sum());stats['signed_sum']+=float(delta.sum());stats['max_abs']=max(stats['max_abs'],float(np.abs(delta).max()));stats['clipped']+=int((angle!=clipped).sum());stats['reversed_adjacent']+=int((np.diff(received,axis=-1)<0).sum());stats['adjacent_pairs']+=received.shape[0]*2

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=('export','stream'),required=True);p.add_argument('--output',type=Path);args=p.parse_args()
 if args.mode=='export':export();return
 assert args.output and not args.output.exists();m.torch.set_num_threads(2);audit=Positions(m.incoming(),args.output,17,256)
 try:
  audit.start()
  for i in range(256):
   audit.row(i)
   if (i+1)%64==0:print(json.dumps(dict(checked=i+1)),file=sys.stderr,flush=True)
  audit.finish();sealed=json.loads((ROOT/'data/provenance/cost-field-full-local-prefix-006-001.json').read_text());audit.ledger.close()
  assert audit.rowhash.hexdigest()==sealed['updates_jsonl_sha256'] and m.sha(args.output.with_suffix('.records.jsonl'))==sealed['checked_record_ledger_sha256']
  bins=[]
  for (channel,index),stats in audit.bins.items():
   n=stats['nodes'];assert n>0
   bins.append(dict(channel=channel,SNR_lower_db=6+2*index,SNR_upper_db=8+2*index,**stats,MAE_coordinate_m=stats['abs_sum']/n,RMSE_coordinate_m=math.sqrt(stats['squared_sum']/n),mean_signed_coordinate_m=stats['signed_sum']/n,clipping_fraction=stats['clipped']/n,reversed_adjacent_fraction=stats['reversed_adjacent']/stats['adjacent_pairs']))
  assert sum(v['frames'] for v in bins)==256 and sum(v['nodes'] for v in bins)==256*1560*3
  result=audit.result('passed_whole_sealed_G256_prefix_coordinate_diagnostic');result.update(bins=bins,position_nodes=sum(v['nodes'] for v in bins),diagnostic_sha256=m.sha(__file__),immutable_report_snapshot_sha256=m.sha(SNAPSHOT),protocol_sha256=m.sha(ROOT/'experiments/cost-field/position-channel-analysis-protocol-006.md'),limitation='Exploratory packet coordinate displacements on an explicit G engineering prefix; not true depth error, calibrated posterior, full training, AP or geometry superiority.')
 except BaseException:
  import traceback
  result=audit.result('failed');result['traceback']=traceback.format_exc();raise
 finally:
  audit.ledger.close();result['checked_record_ledger_sha256']=m.sha(args.output.with_suffix('.records.jsonl'));args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(state=result['state'],position_nodes=result['position_nodes'],bins=[{k:v[k] for k in ('channel','SNR_lower_db','frames','RMSE_coordinate_m','clipping_fraction')} for v in result['bins']])))
if __name__=='__main__':main()
