"""Complete/prefix ordered audit of004 records; bounded local tar stream."""
import argparse, hashlib, io, json, math, struct, subprocess, sys, tarfile, time, traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
EPS=np.finfo(np.float32).eps

def digest(b): return hashlib.sha256(b).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def states(s):
 return {k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=digest(v.contiguous().numpy().tobytes())) for k,v in s.items()}
def safe(b):return torch.load(io.BytesIO(b),map_location='cpu',weights_only=True)
def near(actual,reference,bound,name):
 assert np.shape(actual)==np.shape(reference),name
 error=np.abs(np.asarray(actual,dtype=np.float64)-reference)
 assert np.isfinite(error).all() and (error<=bound).all(),(name,float(error.max(initial=0)),float(np.max(bound)))
 return float(error.max(initial=0))
def blobs(seed,count):
 directory=ROOT/'data/runs'/f'cost-field-full-seed{seed}-004'
 raw_report=(directory/'report.json').read_bytes();report=json.loads(raw_report)
 assert report['seed']==seed and report['update_count']>=count
 terminal=None
 if count==44544:
  assert report['update_count']==44544 and len(report['completed_epochs'])==12 and report['state']=='finished_all44544_full_split_task_updates_pending_actual_terminal_and_complete_proof'
  job=str(report['job_id'])
  accounting=subprocess.check_output(['sacct','-j',job,'--noheader','--parsable2','--format=JobIDRaw,JobID,State,ExitCode'],text=True)
  lines=[x.split('|') for x in accounting.splitlines()]
  assert any(x[0]==job and x[2:4]==['COMPLETED','0:0'] for x in lines),accounting
  assert any(x[0]==job+'.0' and x[2:4]==['COMPLETED','0:0'] for x in lines),accounting
  terminal=dict(raw_job_id=job,sacct=accounting,checked_unix=time.time())
 yield 'metadata.json',json.dumps(dict(seed=seed,count=count,whole=count==44544,terminal=terminal,report_sha256=digest(raw_report))).encode()
 yield 'report.json',raw_report
 yield 'codec-initial.pt',(directory/'codec-initial.pt').read_bytes()
 seen=set();h=hashlib.sha256()
 with (directory/'updates.jsonl').open('rb') as f:
  for index in range(count):
   line=f.readline();assert line.endswith(b'\n');h.update(line);row=json.loads(line)
   yield f'row/{index}.json',line
   frame=row['frame_id']
   if frame not in seen:
    seen.add(frame)
    for folder,ext in [('image_2','.png'),('image_3','.png'),('calib','.txt'),('label_2','.txt')]:
     rel=f'data/kitti/training/{folder}/{frame}{ext}'
     yield rel,(ROOT/rel).read_bytes()
   yield row['artifact'],(ROOT/row['artifact']).read_bytes()
   yield row['optimizer_checkpoint'],(ROOT/row['optimizer_checkpoint']).read_bytes()
  if count==44544:assert not f.read(1)
 if count==44544:
  assert h.hexdigest()==report['updates_jsonl_sha256']
  for epoch in report['completed_epochs']:yield epoch['checkpoint'],(ROOT/epoch['checkpoint']).read_bytes()
 yield 'footer.json',json.dumps(dict(count=count,updates_jsonl_sha256=h.hexdigest(),unique_frames=len(seen))).encode()

def exported(seed,count):
 with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
  for name,b in blobs(seed,count):
   info=tarfile.TarInfo(name);info.size=len(b);archive.addfile(info,io.BytesIO(b))
def incoming():
 with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as archive:
  for member in archive:
   assert member.isfile() and 0<=member.size<=128*1024*1024
   assert not member.name.startswith('/') and '..' not in Path(member.name).parts
   data=archive.extractfile(member).read();assert len(data)==member.size
   yield member.name,data

class Audit:
 def __init__(self,iterator,output,expected_seed,expected_count):
  self.it=iter(iterator);self.out=output;self.seed=expected_seed;self.count=expected_count
  self.checked=0;self.values=0;self.gradvalues=0;self.uses=0;self.ambiguous=0
  self.maxima=dict(Adam_parameter=0.,Adam_moment=0.,baseband=0.,ZF=0.,GT=0.,crop=0.)
  self.source={};self.targets={};self.rowhash=hashlib.sha256();self.noise=[0.,0.,0];self.fading=[0.,0.,0]
  self.ledger=output.with_suffix('.records.jsonl').open('x',buffering=1)
 def get(self,name):
  actual,data=next(self.it);assert actual==name,(actual,name);return data
 def source_bytes(self,name):
  b=self.get(name);item=self.manifest['files'][name]
  assert len(b)==item['bytes'] and digest(b)==item['sha256'],name
  return b
 def start(self):
  self.meta=json.loads(self.get('metadata.json'));assert self.meta['seed']==self.seed and self.meta['count']==self.count and self.meta['whole']==(self.count==44544)
  b=self.get('report.json');assert digest(b)==self.meta['report_sha256'];r=self.r=json.loads(b)
  lockpath=ROOT/'experiments/cost-field/GPU-inputs-004.json';lock=json.loads(lockpath.read_bytes())
  assert r['input_sha256']==sha(lockpath) and r['sources_before']==lock['frozen_inputs'] and r['base_freeze_before']==lock['base_freeze']
  assert r['seed']==self.seed and r['update_count']>=self.count and r['GPU'].find('RTX PRO 6000')>=0 and r['capability']==[12,0]
  assert r['labels_loss_side_only'] and not r['GT_or_labels_or_LiDAR_at_forward_entry'] and r['optimizer_used']
  manifestpath=ROOT/lock['dataset_manifest_path'];assert sha(manifestpath)==lock['dataset_manifest_sha256']==r['dataset_manifest_sha256']
  self.manifest=json.loads(manifestpath.read_bytes());ids=self.manifest['splits']['train']['ids'];assert len(ids)==3712
  orders=[np.random.default_rng(self.seed+1000*epoch).permutation(ids).tolist() for epoch in range(3)]
  rng=np.random.default_rng(self.seed+200000);schedule=[('awgn' if rng.integers(0,2)==0 else 'rayleigh',float(rng.uniform(6,18))) for _ in range(11136)]
  assert r['training_schedule']['per_epoch_order']==orders
  assert r['training_schedule']['channels_and_snr']==[list(x) for x in schedule]
  self.orders,self.schedule=orders,schedule
  raw=torch.load(ROOT/'assets/liga-native-001/liga-author-download',map_location='cpu',weights_only=True)['model_state']
  assert sha(ROOT/'assets/liga-native-001/liga-author-download')=='3f36af30b686f23d0868482675177b29cc50c2b378577dd12f9092ddb0235b2e'
  for k in r['sparse_layout_converted']:raw[k]=raw[k].permute(4,0,1,2,3).contiguous()
  author=states(raw);assert len(author)==484 and author==r['loaded_states'];del raw
  self.author_hash=digest(json.dumps(author,sort_keys=True).encode());self.author=author
  b=self.get('codec-initial.pt');assert digest(b)==r['codec_initial_checkpoint_sha256']
  self.initial=safe(b);self.previous={arm:dict(codec=s,optimizer=dict(state={})) for arm,s in self.initial.items()}
  assert {arm:states(s) for arm,s in self.initial.items()}==r['codec_initial_states']
  assert r['codec_parameters_per_arm']==dict(G=1650,P=1650,S=1650,B=1751)
  for arm in self.initial:
   axis=self.initial[arm]['depth_axis'].numpy();assert len(axis)==72 and abs(axis[0]-2)<1e-6 and abs(axis[-1]-59.6)<4e-6
   assert np.array_equal(axis,np.asarray(r['depth_coordinate_contract']['voxel_query_axis'],np.float32))
  if self.meta['whole']:
   assert r['update_count']==44544 and len(r['completed_epochs'])==12 and r['final_states']==author and self.meta['terminal']
   assert r['base_freeze_after']==lock['base_freeze'] and r['sources_after']==lock['frozen_inputs'] and r['sensor_inputs_after']==lock['sensor_inputs']
   assert r['all_parameters_frozen'] and r['all_parameter_gradients_absent']
 def read_source(self,frame):
  if frame in self.source:return self.source[frame]
  shape=None
  for folder in ('image_2','image_3'):
   b=self.source_bytes(f'data/kitti/training/{folder}/{frame}.png')
   assert b[:8]==b'\x89PNG\r\n\x1a\n' and b[12:16]==b'IHDR'
   w,h=struct.unpack('>II',b[16:24]);current=(h,w)
   if shape is None:shape=current
   else:assert shape==current
  calib=self.source_bytes(f'data/kitti/training/calib/{frame}.txt').decode()
  projections={line.split(':')[0]:np.asarray(line.split(':')[1].split(),np.float32).reshape(3,4) for line in calib.splitlines() if line.startswith(('P2:','P3:'))}
  label=self.source_bytes(f'data/kitti/training/label_2/{frame}.txt');camera=[];classes=[]
  for line in label.decode().splitlines():
   x=line.split();cls={'Van':'Car','Person_sitting':'Pedestrian'}.get(x[0],x[0])
   if cls not in ('Car','Pedestrian','Cyclist'):continue
   camera.append([float(v) for v in (x[11],x[12],x[13],x[10],x[8],x[9],x[14])]);classes.append(('Car','Pedestrian','Cyclist').index(cls)+1)
  self.source[frame]=dict(shape=shape,projections=projections,camera=np.asarray(camera,np.float32).reshape(-1,7),classes=np.asarray(classes,np.float32),labelsha=digest(label))
  return self.source[frame]
 def gt(self,a,source):
  camera=source['camera'];assert np.array_equal(camera,a['camera_gt'])
  c=camera.astype(np.float64);lidar=np.empty_like(c)
  lidar[:,:3]=np.column_stack((c[:,2],-c[:,0],-c[:,1]+c[:,4]/2))
  lidar[:,3:6]=c[:,[3,5,4]];lidar[:,6]=-(c[:,6]+math.pi/2)
  template=np.array([[x,y,z] for x in (-.5,.5) for y in (-.5,.5) for z in (-.5,.5)])
  v=lidar[:,None,3:6]*template;co=np.cos(lidar[:,6,None]);si=np.sin(lidar[:,6,None])
  corners=np.stack((v[:,:,0]*co-v[:,:,1]*si,v[:,:,0]*si+v[:,:,1]*co,v[:,:,2]),-1)+lidar[:,None,:3]
  low=np.array([2.,-30.4,-3.]);high=np.array([59.6,30.4,1.])
  envelope=64*EPS*(1+np.abs(lidar[:,:3])+lidar[:,3:6].sum(1)[:,None]+np.maximum(np.abs(low),np.abs(high)))
  robust=((corners-envelope[:,None]>=low)&(corners+envelope[:,None]<=high)).all(2).any(1)
  possible=((corners+envelope[:,None]>=low)&(corners-envelope[:,None]<=high)).all(2).any(1)
  mask=a['range_keep_mask'];assert mask.dtype==np.bool_ and mask.shape==robust.shape
  assert (mask[robust]).all() and (~mask[~possible]).all();self.ambiguous+=int((robust!=possible).sum())
  expected=np.column_stack((lidar[mask],source['classes'][mask]))
  self.maxima['GT']=max(self.maxima['GT'],near(a['gt_boxes'],expected,64*EPS*(1+np.abs(expected)),'GT'))
  assert np.array_equal(a['image_shape'],np.asarray(source['shape'],np.int32)[None])
  offsets=np.asarray([0,max(source['shape'][0]-320,0)])
  assert np.array_equal(a['crop_offsets'],offsets)
  original=source['projections'];K=original['P2'][:,:3].astype(np.float64);cropped=K.copy();cropped[0,2]-=offsets[0];cropped[1,2]-=offsets[1]
  for view in ('P2','P3'):
   assert np.array_equal(a['original_'+view],original[view])
   reference=cropped@np.linalg.inv(K)@original[view].astype(np.float64)
   self.maxima['crop']=max(self.maxima['crop'],near(a['cropped_'+view],reference,64*EPS*(1+np.abs(reference)+np.abs(original[view])),'crop'))
 def physics(self,a,row):
  tx=a['wire_tx'];rx=a['wire_received'];h=a['channel_fading'].astype(np.float64);n=a['channel_noise'].astype(np.float64);y=a['channel_received_baseband'].astype(np.float64)
  assert tx.shape==rx.shape==h.shape==n.shape==y.shape==(1,29640,2) and row['complex_uses']==29640
  x=tx.astype(np.float64);packet=x.reshape(-1,19,2)
  if row['arm']=='B':groups=[(packet[:,:15],15),(packet[:,15:],4)]
  else:groups=[(packet[:,k:k+1],1) for k in range(3)]+[(packet[:,3+4*k:7+4*k],4) for k in range(3)]+[(packet[:,15:],4)]
  for group,energy in groups:assert (np.abs((group*group).sum((1,2))-energy)<=energy*1e-6).all()
  energy=float((x*x).sum());assert abs(energy-row['energy'])<=1e-8 and abs(energy-29640)<=29640e-6
  if row['channel']=='awgn':
   assert np.array_equal(h[...,0],np.ones(h.shape[:-1])) and not h[...,1].any()
   ref=x+n;bound=64*EPS*(np.abs(x)+np.abs(n)+np.finfo(np.float32).tiny)
   self.maxima['baseband']=max(self.maxima['baseband'],near(y,ref,bound,'AWGN'))
   assert np.array_equal(a['wire_received'],a['channel_received_baseband'])
  else:
   hr,hi=h[...,0],h[...,1];xr,xi=x[...,0],x[...,1]
   ref=np.stack((hr*xr-hi*xi+n[...,0],hr*xi+hi*xr+n[...,1]),-1)
   mag=np.stack((np.abs(hr*xr)+np.abs(hi*xi)+np.abs(n[...,0]),np.abs(hr*xi)+np.abs(hi*xr)+np.abs(n[...,1])),-1)
   self.maxima['baseband']=max(self.maxima['baseband'],near(y,ref,64*EPS*(mag+np.finfo(np.float32).tiny),'Rayleigh baseband'))
   denom=hr*hr+hi*hi;assert (denom>0).all()
   ref=np.stack(((hr*y[...,0]+hi*y[...,1])/denom,(hr*y[...,1]-hi*y[...,0])/denom),-1)
   mag=np.stack((np.abs(hr*y[...,0])+np.abs(hi*y[...,1]),np.abs(hr*y[...,1])+np.abs(hi*y[...,0])),-1)/denom[...,None]
   self.maxima['ZF']=max(self.maxima['ZF'],near(rx,ref,64*EPS*(mag+np.abs(ref)+np.finfo(np.float32).tiny),'ZF'))
   self.fading[0]+=float(h.sum());self.fading[1]+=float((h*h).sum());self.fading[2]+=h.size
  normalized=n/math.sqrt(10**(-row['snr_db']/10)/2)
  self.noise[0]+=float(normalized.sum());self.noise[1]+=float((normalized*normalized).sum());self.noise[2]+=n.size
  self.uses+=29640
 def optimizer(self,a,row,checkpoint):
  arm=row['arm'];old=self.previous[arm];saved=checkpoint['codec'];optimizer=checkpoint['optimizer']
  assert states(old['codec'])==row['codec_states_before'] and states(saved)==row['codec_states_after']
  changed=[k for k in row['codec_states_before'] if row['codec_states_before'][k]!=row['codec_states_after'][k]];assert changed==row['changed_states']
  assert torch.equal(saved['depth_axis'],self.initial[arm]['depth_axis'])
  names=[k for k in old['codec'] if k!='depth_axis'];assert set('codec_gradient_'+k for k in names)=={k for k in a if k.startswith('codec_gradient_')}
  assert set('clipped_gradient_'+k for k in names)=={k for k in a if k.startswith('clipped_gradient_')}
  groups=optimizer['param_groups'];assert len(groups)==1;group=groups[0]
  assert group['lr']==1e-3 and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==0
  assert not any(group.get(k,False) for k in ('amsgrad','maximize','capturable','differentiable'))
  assert group.get('fused') in (None,False) and len(group['params'])==len(names)
  assert len(optimizer['state'])==len(names) and set(optimizer['state'])==set(group['params'])
  norm=float(np.sqrt(sum(np.square(a['codec_gradient_'+k].astype(np.float64)).sum() for k in names)))
  assert abs(norm-row['gradient_norm_before_clip'])<=64*EPS*(1+norm)
  coeff=min(1.,10/(row['gradient_norm_before_clip']+1e-6));t=row['step']+1
  assert sum(a['codec_gradient_'+k].size for k in names)==self.r['codec_parameters_per_arm'][arm]
  for index,name in zip(group['params'],names):
   g=a['codec_gradient_'+name].astype(np.float64);gg=a['clipped_gradient_'+name].astype(np.float64)
   near(gg,g*coeff,8*EPS*np.abs(g*coeff)+EPS,'clip')
   moment=optimizer['state'][index];previous=old['optimizer']['state'].get(index)
   assert set(moment)=={'step','exp_avg','exp_avg_sq'} and float(moment['step'])==t
   prevm=np.zeros_like(gg) if previous is None else previous['exp_avg'].numpy().astype(np.float64)
   prevv=np.zeros_like(gg) if previous is None else previous['exp_avg_sq'].numpy().astype(np.float64)
   if previous is not None:assert float(previous['step'])==t-1
   m=.9*prevm+.1*gg;v=.999*prevv+.001*gg*gg
   for key,reference,bound in [('exp_avg',m,128*EPS*(.9*np.abs(prevm)+.1*np.abs(gg)+np.finfo(np.float32).tiny)),('exp_avg_sq',v,128*EPS*(.999*np.abs(prevv)+.001*gg*gg+np.finfo(np.float32).tiny))]:
    assert moment[key].dtype==torch.float32 and torch.isfinite(moment[key]).all()
    self.maxima['Adam_moment']=max(self.maxima['Adam_moment'],near(moment[key].numpy(),reference,bound,key))
   parameter=old['codec'][name].numpy().astype(np.float64)
   delta=1e-3/(1-.9**t)*m/(np.sqrt(v/(1-.999**t))+1e-8);reference=parameter-delta
   self.maxima['Adam_parameter']=max(self.maxima['Adam_parameter'],near(saved[name].numpy(),reference,128*EPS*(1+np.abs(parameter)+np.abs(delta)+np.abs(reference)),name))
   self.gradvalues+=g.size+gg.size
  self.previous[arm]=checkpoint
 def row(self,index):
  line=self.get(f'row/{index}.json');self.rowhash.update(line);row=json.loads(line)
  arm=('G','P','S','B')[index//11136];step=index%11136;epoch=step//3712;position=step%3712;frame=self.orders[epoch][position]
  assert (row['arm'],row['step'],row['epoch'],row['epoch_index'],row['frame_id'])==(arm,step,epoch+1,position,frame)
  channel,snr=self.schedule[step];assert row['channel']==channel and row['snr_db']==snr
  source=self.read_source(frame);assert row['label_file']==f'data/kitti/training/label_2/{frame}.txt' and row['label_sha256']==source['labelsha']
  stem=f'data/runs/cost-field-full-seed{self.seed}-004/updates/{arm}/epoch{epoch+1}/block{step//256:04d}/{frame}-step{step}'
  assert row['artifact']==stem+'.npz' and row['optimizer_checkpoint']==stem+'-state.pt'
  b=self.get(row['artifact']);assert len(b)==row['bytes'] and digest(b)==row['sha256']
  with np.load(io.BytesIO(b),allow_pickle=False) as archive:
   assert set(archive.files)==set(row['arrays'])
   a={k:archive[k] for k in archive.files}
  for k,v in a.items():
   assert np.isfinite(v).all() and dict(dtype=str(v.dtype),shape=list(v.shape),sha256=digest(np.ascontiguousarray(v).tobytes()))==row['arrays'][k],k
   self.values+=v.size
  assert row['readonly_states_sha256']==self.author_hash and row['all_parameter_gradients_absent'] and row['cut_calls']==1 and row['forbidden_calls']==0
  assert row['calls']==dict(detector=1,image_backbone=2,feature_neck=2,build_cost=1,head3D=1,depth_logits=1,forbidden=0)
  assert set(row['inference_keys'])=={'batch_size','left_img','right_img','calib','image_shape','frame_id'}
  assert row['received_only_decode_identical_after_private_clear']
  assert a['pred_boxes'].shape==(row['prediction_count'],7) and a['pred_scores'].shape==a['pred_labels'].shape==(row['prediction_count'],)
  self.gt(a,source);self.physics(a,row)
  loss=a['losses'];assert loss.shape==(3,) and (loss>=0).all() and loss[0]==row['cls_stats']['rpn_loss_cls']
  box=np.float32(row['box_stats']['rpn_loss_loc'])
  for k in ('rpn_loss_iou','rpn_loss_dir'):box=np.float32(box+np.float32(row['box_stats'][k]))
  assert loss[1]==float(box) and loss[2]==float(np.float32(np.float32(loss[0])+box))
  assert int((a['assigned_box_cls_labels']>0).sum())==row['positive_anchors']
  targets={k:v for k,v in row['arrays'].items() if k.startswith('assigned_')}
  if frame in self.targets:assert targets==self.targets[frame]
  else:self.targets[frame]=targets
  b=self.get(row['optimizer_checkpoint']);assert digest(b)==row['optimizer_checkpoint_sha256'];self.optimizer(a,row,safe(b))
  assert {k:int(np.count_nonzero(v)) for k,v in a.items() if k.startswith('codec_gradient_')}==row['codec_gradient_nonzero']
  self.ledger.write(json.dumps(dict(index=index,arm=arm,step=step,frame=frame,artifact=row['artifact'],sha256=row['sha256'],optimizer_checkpoint=row['optimizer_checkpoint'],optimizer_checkpoint_sha256=row['optimizer_checkpoint_sha256'],row_sha256=digest(line),source_label_sha256=source['labelsha'],positive_anchors=row['positive_anchors'],losses=loss.tolist()),allow_nan=False)+'\n')
  self.checked+=1
 def finish(self):
  if self.meta['whole']:
   assert {arm:states(s['codec']) for arm,s in self.previous.items()}==self.r['codec_final_states']
   assert all(states(self.previous[arm]['codec'])!=self.r['codec_initial_states'][arm] for arm in self.previous)
   assert [(x['arm'],x['epoch'],x['updates']) for x in self.r['completed_epochs']]==[(a,e,3712) for a in ('G','P','S','B') for e in (1,2,3)]
   for epoch in self.r['completed_epochs']:
    b=self.get(epoch['checkpoint']);assert digest(b)==epoch['sha256']
   assert self.r['updates_jsonl_sha256']==self.rowhash.hexdigest() and len(self.source)==3712
  footer=json.loads(self.get('footer.json'));assert footer==dict(count=self.count,updates_jsonl_sha256=self.rowhash.hexdigest(),unique_frames=len(self.source))
  try:next(self.it)
  except StopIteration:pass
  else:raise AssertionError('Unexpected stream suffix')
 def result(self,state):
  return dict(state=state,seed=self.seed,job_id=self.r['job_id'] if hasattr(self,'r') else None,whole_training_closed=self.count==44544 and self.checked==44544 and state=='passed_complete_training_stream',requested_updates=self.count,checked_updates=self.checked,unique_frames=len(self.source),complete_saved_array_values=self.values,gradient_values=self.gradvalues,complex_uses=self.uses,range_ambiguous_decisions=self.ambiguous,max_errors=self.maxima,noise_normalized_moments=self.noise,fading_component_moments=self.fading,verifier_sha256=sha(__file__),protocol_sha256=sha(ROOT/'experiments/cost-field/full-proof-protocol-005.md'),report_sha256=self.meta['report_sha256'] if hasattr(self,'meta') else None,updates_jsonl_sha256=self.rowhash.hexdigest(),actual_terminal=self.meta.get('terminal') if hasattr(self,'meta') else None,checked_unix=time.time(),limitation='Full saved-record transitions, not independent CUDA gradients, full decoded fields or processed RGB replay. Repeated targets checked for identity, not independent assignment. Prefix is engineering only; no validation AP evidence.')

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=('export','native','stream'),required=True);p.add_argument('--seed',type=int,choices=(17,23,41),required=True);p.add_argument('--count',type=int,default=44544);p.add_argument('--output',type=Path)
 args=p.parse_args();assert 1<=args.count<=44544
 if args.mode=='export':exported(args.seed,args.count);return
 assert args.output and not args.output.exists();torch.set_num_threads(2)
 audit=Audit(incoming() if args.mode=='stream' else blobs(args.seed,args.count),args.output,args.seed,args.count)
 try:
  audit.start()
  for index in range(args.count):
   audit.row(index)
   if (index+1)%64==0:print(json.dumps(dict(checked=index+1,count=args.count,seed=args.seed)),file=sys.stderr,flush=True)
  audit.finish();result=audit.result('passed_complete_training_stream' if args.count==44544 else 'passed_explicit_live_prefix_engineering_only')
 except BaseException:
  result=audit.result('failed');result['traceback']=traceback.format_exc();raise
 finally:
  audit.ledger.close();result['checked_record_ledger_sha256']=sha(args.output.with_suffix('.records.jsonl'))
  args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:result[k] for k in ('state','seed','checked_updates','complete_saved_array_values','complex_uses','whole_training_closed')}))
if __name__=='__main__':main()
