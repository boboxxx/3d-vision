"""Full3769 native writer/wire closure and bounded complete local wire proof."""
import argparse,hashlib,importlib.util,io,json,os,subprocess,sys,tarfile,tempfile,time,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'data/provenance/audit-cost-field-full-006.py'
spec=importlib.util.spec_from_file_location('training_audit006',BASE);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
WRITER=ROOT/'experiments/cost-field/validation_code_005/writer_replay_006.py'
spec=importlib.util.spec_from_file_location('writer006',WRITER);writer=importlib.util.module_from_spec(spec);spec.loader.exec_module(writer)
sha=base.sha;digest=base.digest;safe=base.safe
INDEX=ROOT/'data/provenance/cost-field-val-source-index-007.json'
MANIFEST=ROOT/'data/provenance/cost-field-full-dataset-manifest-004.json'


def payload(run,mode):
 directory=ROOT/'data/runs'/run;reportbytes=(directory/'report.json').read_bytes();r=json.loads(reportbytes)
 assert r['state']=='finished_full3769_final_epoch_inference_pending_terminal_wire_text_AP_closure' and r['frames_complete']==3769
 job=str(r['job_id']);terminal=None
 if mode=='native':
  accounting=subprocess.check_output(['sacct','-j',job,'--noheader','--parsable2','--format=JobIDRaw,JobID,State,ExitCode'],text=True)
  lines=[x.split('|') for x in accounting.splitlines()]
  assert any(x[0]==job and x[2:4]==['COMPLETED','0:0'] for x in lines)
  assert any(x[0]==job+'.0' and x[2:4]==['COMPLETED','0:0'] for x in lines)
  terminal=dict(actual_terminal=True,job_id=job,sacct=accounting,report_sha256=digest(reportbytes))
 else:
  terminal=json.loads((directory/'terminal.json').read_text());assert terminal['actual_terminal'] and terminal['report_sha256']==digest(reportbytes)
 yield 'metadata.json',json.dumps(dict(run=run,terminal=terminal,source_index_sha256=sha(INDEX))).encode()
 yield 'report.json',reportbytes
 trainingdir=ROOT/'data/runs'/f"cost-field-full-seed{r['seed']}-004"
 yield 'training-report.json',(trainingdir/'report.json').read_bytes()
 for index,p in enumerate(r['training_proofs']):yield f'training-proof-{index}.json',Path(p['path']).read_bytes()
 yield 'final-codec.pt',(ROOT/r['final_codec_checkpoint']).read_bytes()
 h=hashlib.sha256()
 with (directory/'frames.jsonl').open('rb') as f:
  for index,frame in enumerate(r['validation_ids']):
   line=f.readline();assert line.endswith(b'\n');h.update(line);row=json.loads(line)
   yield f'row/{index}.json',line
   cal=f'data/kitti/training/calib/{frame}.txt';yield cal,(ROOT/cal).read_bytes()
   yield row['artifact'],(ROOT/row['artifact']).read_bytes()
   yield row['prediction'],(ROOT/row['prediction']).read_bytes()
  assert not f.read(1)
 assert h.hexdigest()==r['frames_jsonl_sha256']
 yield 'footer.json',json.dumps(dict(frames=3769,frames_jsonl_sha256=h.hexdigest())).encode()

def export(run):
 with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
  for name,data in payload(run,'export'):
   info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data))

class Validation:
 def __init__(self,iterator,run,mode,out):
  self.it=iter(iterator);self.run=run;self.mode=mode;self.out=out;self.checked=0;self.values=0;self.uses=0;self.written=0;self.noise=[0.,0.,0];self.fading=[0.,0.,0]
  self.maxima=dict(baseband=0.,ZF=0.,crop=0.);self.rowhash=hashlib.sha256();self.ledger=out.with_suffix('.records.jsonl').open('x',buffering=1)
  self.directory=out.with_suffix('.small-artifacts');assert not self.directory.exists();self.directory.mkdir();(self.directory/'data').mkdir()
  self.writer_sources=None
 def get(self,name):
  actual,data=next(self.it);assert actual==name,(actual,name);return data
 def retain(self,name,b):
  p=self.directory/name;p.write_bytes(b);return p
 def start(self):
  self.meta=json.loads(self.get('metadata.json'));assert self.meta['run']==self.run and self.meta['source_index_sha256']==sha(INDEX)
  b=self.get('report.json');self.r=r=json.loads(b);self.retain('report.json',b)
  self.terminal=self.meta['terminal'];assert self.terminal['actual_terminal'] and self.terminal['job_id']==r['job_id'] and self.terminal['report_sha256']==digest(b)
  job=str(r['job_id']);lines=[x.split('|') for x in self.terminal['sacct'].splitlines()]
  assert any(x[0]==job and x[2:4]==['COMPLETED','0:0'] for x in lines) and any(x[0]==job+'.0' and x[2:4]==['COMPLETED','0:0'] for x in lines)
  self.retain('terminal.json',json.dumps(self.terminal,indent=2).encode())
  assert r['state']=='finished_full3769_final_epoch_inference_pending_terminal_wire_text_AP_closure' and r['frames_complete']==3769 and not r['AP_computed'] and not r['labels_GT_LiDAR_at_inference']
  assert r['inference_noise_seed_rule']=='2027100000+integer(frame_id)'
  assert r['arm'] in ('G','P','S','B') and r['seed'] in (17,23,41)
  assert r['channel'] in ('identity','awgn','rayleigh') and r['snr_db'] in (6,8,10,12,14,16,18)
  assert self.run==f"cost-field-val-seed{r['seed']}-{r['arm']}-{r['channel']}-{r['snr_db']}-005"
  assert r['channel']!='identity' or r['snr_db']==10
  lockpath=ROOT/'experiments/cost-field/GPU-inputs-004.json';lock=json.loads(lockpath.read_text())
  assert r['sources_before']==r['sources_after']==lock['frozen_inputs'] and r['base_freeze_before']==r['base_freeze_after']==lock['base_freeze']
  assert r['inference_sources_before']==r['inference_sources_after']
  for p,h in r['inference_sources_before'].items():assert sha(ROOT/p)==h,p
  self.index=json.loads(INDEX.read_text());self.manifest=json.loads(MANIFEST.read_text())
  assert sha(MANIFEST)==r['manifest_sha256']==self.index['manifest_sha256'] and self.index['state']=='closed_public_source_shapes_all3769_and_six_original_caches'
  self.ids=self.manifest['splits']['val']['ids'];assert self.ids==r['validation_ids']==self.index['ordered_ids'] and len(self.ids)==3769
  b=self.get('training-report.json');assert digest(b)==r['training_report_sha256'];training=json.loads(b);self.retain('training-report.json',b)
  assert training['update_count']==44544 and training['state']=='finished_all44544_full_split_task_updates_pending_actual_terminal_and_complete_proof' and training['seed']==r['seed']
  proofs=[]
  for i,p in enumerate(r['training_proofs']):
   b=self.get(f'training-proof-{i}.json');assert digest(b)==p['sha256'];proof=json.loads(b);self.retain(f'training-proof-{i}.json',b)
   assert proof['state']=='passed_complete_training_stream' and proof['whole_training_closed'] and proof['checked_updates']==44544 and proof['seed']==r['seed']
   assert proof['report_sha256']==r['training_report_sha256'] and proof['updates_jsonl_sha256']==training['updates_jsonl_sha256'] and proof['actual_terminal']['raw_job_id']==training['job_id']
   proofs.append(proof)
  assert len(proofs)==2 and proofs[0]['checked_record_ledger_sha256']==proofs[1]['checked_record_ledger_sha256']
  checkpoint=[x for x in training['completed_epochs'] if x['arm']==r['arm'] and x['epoch']==3];assert len(checkpoint)==1
  assert r['final_codec_checkpoint']==checkpoint[0]['checkpoint'] and r['final_codec_checkpoint_sha256']==checkpoint[0]['sha256']
  b=self.get('final-codec.pt');assert digest(b)==r['final_codec_checkpoint_sha256'];state=safe(b)['codec'];self.retain('final-codec.pt',b)
  assert base.states(state)==training['codec_final_states'][r['arm']]==r['codec_initial_states']==r['codec_final_states']
  raw=torch.load(ROOT/'assets/liga-native-001/liga-author-download',map_location='cpu',weights_only=True)['model_state']
  for k in training['sparse_layout_converted']:raw[k]=raw[k].permute(4,0,1,2,3).contiguous()
  author=base.states(raw);assert len(author)==484 and author==training['loaded_states']==training['final_states']==r['author_initial_states']==r['author_final_states'];del raw
  self.authorhash=digest(json.dumps(author,sort_keys=True).encode());self.codechash=digest(json.dumps(base.states(state),sort_keys=True).encode())
  if self.mode=='native':
   _,_,self.writer_sources=writer.load(ROOT)
   assert all(r['sources_before'][p]==h for p,h in self.writer_sources.items())
   assert all(sha(ROOT/p)==h for p,h in r['sources_before'].items())
  else:
   assert self.terminal['state']=='closed_actual_full3769_inference_wire_native_writer_pending_official_AP'
   assert self.terminal['frames']==self.terminal['native_writer_frames']==3769
 def crop(self,a,cal,source):
  parsed={line.split(':')[0]:np.asarray(line.split(':')[1].split(),np.float32) for line in cal.decode().splitlines() if ':' in line}
  P2=parsed['P2'].reshape(3,4);P3=parsed['P3'].reshape(3,4)
  assert np.array_equal(a['original_P2'],P2) and np.array_equal(a['original_P3'],P3)
  shape=source['shape'];assert np.array_equal(a['image_shape'],np.asarray(shape,np.int32)[None])
  offset=np.asarray([0,max(shape[0]-320,0)]);assert np.array_equal(a['crop_offsets'],offset)
  K=P2[:,:3].astype(np.float64);cropped=K.copy();cropped[0,2]-=offset[0];cropped[1,2]-=offset[1]
  for key,P in [('P2',P2),('P3',P3)]:
   reference=cropped@np.linalg.inv(K)@P.astype(np.float64)
   self.maxima['crop']=max(self.maxima['crop'],base.near(a['cropped_'+key],reference,64*base.EPS*(1+np.abs(reference)+np.abs(P)),'crop'))
 def row(self,i):
  line=self.get(f'row/{i}.json');self.rowhash.update(line);row=json.loads(line);frame=self.ids[i];r=self.r
  assert (row['index'],row['frame_id'],row['arm'],row['seed'],row['channel'],row['snr_db'])==(i,frame,r['arm'],r['seed'],r['channel'],r['snr_db'])
  source=self.index['frames'][frame]
  for folder in ('image_2','image_3'):
   item=self.manifest['files'][f'data/kitti/training/{folder}/{frame}.png'];assert row['source'][folder]==item and item['sha256']==source['source_png_sha256'][folder]
  calpath=f'data/kitti/training/calib/{frame}.txt';cal=self.get(calpath);item=self.manifest['files'][calpath]
  assert len(cal)==item['bytes'] and digest(cal)==item['sha256'] and row['source']['calib']==item
  prefix='data/runs/'+self.run;assert row['artifact']==f'{prefix}/wires/block{i//256:04d}/{frame}.npz' and row['prediction']==f'{prefix}/data/{frame}.txt'
  b=self.get(row['artifact']);assert len(b)==row['bytes'] and digest(b)==row['sha256']
  with np.load(io.BytesIO(b),allow_pickle=False) as f:
   assert set(f.files)==set(row['arrays']);a={k:f[k] for k in f.files}
  expected={'original_P2','original_P3','image_shape','cropped_P2','cropped_P3','crop_offsets','pred_boxes','pred_scores','pred_labels','wire_tx','wire_received','channel_noise','channel_fading','channel_received_baseband'}
  assert set(a)==expected
  for k,v in a.items():
   assert np.isfinite(v).all() and dict(dtype=str(v.dtype),shape=list(v.shape),sha256=digest(np.ascontiguousarray(v).tobytes()))==row['arrays'][k],k
   self.values+=v.size
  assert row['original_states_sha256']==self.authorhash and row['codec_states_sha256']==self.codechash and row['received_only_decode_identical_after_private_clear']
  assert row['calls']==dict(detector=1,image_backbone=2,feature_neck=2,build_cost=1,head3D=1,depth_logits=1,forbidden=0,cut=1)
  assert set(row['inference_keys'])=={'batch_size','left_img','right_img','calib','image_shape','frame_id'}
  self.crop(a,cal,source)
  for key,shape in [('decoded_cost',[1,32,72,*row['cost_hw']]),('decoded_appearance',[1,32,*row['appearance_hw']])]:
   assert row[key]['shape']==shape and row[key]['dtype']=='float32' and len(row[key]['sha256'])==64
  assert row['cost_hw']==row['appearance_hw'] and row['complex_uses']==(row['cost_hw'][0]//4)*(row['cost_hw'][1]//4)*19
  if row['channel']=='identity':
   assert np.array_equal(a['wire_tx'],a['wire_received']) and np.array_equal(a['wire_tx'],a['channel_received_baseband']) and not a['channel_noise'].any()
   copy=dict(row,channel='awgn');base.Audit.physics(self,a,copy)
  else:base.Audit.physics(self,a,row)
  text=self.get(row['prediction']);assert digest(text)==row['prediction_sha256'];self.retain('data/'+frame+'.txt',text)
  lines=text.decode().splitlines();assert len(lines)==row['prediction_count']==len(a['pred_boxes'])
  assert a['pred_boxes'].shape==(len(lines),7) and a['pred_scores'].shape==a['pred_labels'].shape==(len(lines),)
  assert a['pred_labels'].dtype.kind in ('i','u') and ((a['pred_labels']>=1)&(a['pred_labels']<=3)).all()
  assert (a['pred_boxes'][:,3:6]>0).all() and ((a['pred_scores']>=0)&(a['pred_scores']<=1)).all()
  for j,x in enumerate(lines):
   fields=x.split();assert len(fields)==16 and fields[0]==('Car','Pedestrian','Cyclist')[a['pred_labels'][j]-1]
   assert np.isfinite([float(v) for v in fields[1:]]).all() and fields[1:3]==['-1','-1']
   assert abs(float(fields[15])-float(a['pred_scores'][j]))<=.5e-8+1e-15
  if self.mode=='native':
   with tempfile.TemporaryDirectory(prefix='author-writer-replay-') as temp:
    assert writer.replay(ROOT,a,frame,Path(temp))==text
   self.written+=1
  ledger=dict(index=i,frame_id=frame,artifact=row['artifact'],artifact_sha256=row['sha256'],prediction_sha256=row['prediction_sha256'],prediction_count=row['prediction_count'],row_sha256=digest(line),noise_sha256=row['arrays']['channel_noise']['sha256'],fading_sha256=row['arrays']['channel_fading']['sha256'],complex_uses=row['complex_uses'],energy=row['energy'])
  self.ledger.write(json.dumps(ledger,allow_nan=False)+'\n');self.checked+=1
 def finish(self):
  footer=json.loads(self.get('footer.json'));assert footer==dict(frames=3769,frames_jsonl_sha256=self.rowhash.hexdigest()) and footer['frames_jsonl_sha256']==self.r['frames_jsonl_sha256']
  try:next(self.it)
  except StopIteration:pass
  else:raise AssertionError('Unexpected stream suffix')
  assert self.checked==3769 and {p.stem for p in (self.directory/'data').glob('*.txt')}==set(self.ids)
  self.ledger.close()
  if self.mode!='native':assert sha(self.out.with_suffix('.records.jsonl'))==self.terminal['checked_record_ledger_sha256']
 def result(self,state):
  return dict(state=state,actual_terminal=True,run=self.run,job_id=self.r['job_id'] if hasattr(self,'r') else None,report_sha256=digest((self.directory/'report.json').read_bytes()) if (self.directory/'report.json').exists() else None,
   sacct=self.terminal['sacct'] if hasattr(self,'terminal') else None,frames=self.checked,native_writer_frames=self.written,complete_saved_array_values=self.values,complex_uses=self.uses,max_errors=self.maxima,
   noise_normalized_moments=self.noise,fading_component_moments=self.fading,frames_jsonl_sha256=self.rowhash.hexdigest(),source_index_sha256=sha(INDEX),manifest_sha256=sha(MANIFEST),verifier_sha256=sha(__file__),base_auditor_sha256=sha(BASE),writer_helper_sha256=sha(WRITER),writer_sources=self.writer_sources,checked_unix=time.time(),AP_endpoint_closed=False,
   limitation='Full saved wire/text proof and native author-writer replay; local source image shapes/hashes use closed original cache lineage. No independent processed RGB/decoded-field/CUDA-gradient replay, no AP closure.')

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=('native','export','stream'),required=True);p.add_argument('--run',required=True);p.add_argument('--output',type=Path)
 args=p.parse_args();assert '/' not in args.run and args.run.startswith('cost-field-val-seed')
 if args.mode=='export':export(args.run);return
 out=args.output or ROOT/'data/runs'/args.run/'terminal.json';assert not out.exists();torch.set_num_threads(2)
 audit=Validation(base.incoming() if args.mode=='stream' else payload(args.run,'native'),args.run,args.mode,out)
 try:
  audit.start()
  for i in range(3769):
   audit.row(i)
   if (i+1)%128==0:print(json.dumps(dict(checked=i+1,run=args.run)),file=sys.stderr,flush=True)
  audit.finish();result=audit.result('closed_actual_full3769_inference_wire_native_writer_pending_official_AP' if args.mode=='native' else 'passed_complete3769_wire_stream_and_fixed_text')
 except BaseException:result=audit.result('failed');result['traceback']=traceback.format_exc();raise
 finally:
  audit.ledger.close();result['checked_record_ledger_sha256']=sha(out.with_suffix('.records.jsonl'));out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:result[k] for k in ('state','frames','native_writer_frames','complete_saved_array_values','complex_uses')}))
if __name__=='__main__':main()
