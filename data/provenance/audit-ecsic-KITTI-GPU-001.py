"""Complete two-step RGB/RD/gradient/Adam audit, with bounded tar replay."""
import argparse,hashlib,io,json,math,subprocess,sys,tarfile,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'data/engineering/ecsic-KITTI-GPU-001'
EPS=np.finfo(np.float32).eps
def digest(b):return hashlib.sha256(b).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def descriptor(v):
 a=np.ascontiguousarray(v)
 return dict(dtype=str(a.dtype),shape=list(a.shape),sha256=digest(a.tobytes()))
def states(s):return {k:descriptor(v.contiguous().numpy()) for k,v in s.items()}
def safe(b):return torch.load(io.BytesIO(b),map_location='cpu',weights_only=True)
def near(actual,reference,bound,name):
 e=np.abs(np.asarray(actual,dtype=np.float64)-reference)
 assert np.shape(actual)==np.shape(reference) and np.isfinite(e).all() and (e<=bound).all(),(name,float(e.max(initial=0)),float(np.max(bound)))
 return float(e.max(initial=0))
def blobs():
 b=(RUN/'report.json').read_bytes();r=json.loads(b)
 assert r['state']=='finished_two_native_RD_updates_pending_actual_terminal_and_complete_audit' and len(r['steps'])==2
 job=str(r['job_id']);accounting=subprocess.check_output(['sacct','-j',job,'--format=JobIDRaw,State,ExitCode','-n','-P'],text=True)
 rows=[x.split('|') for x in accounting.splitlines()]
 for identity in (job,job+'.0'):assert any(x==[identity,'COMPLETED','0:0'] for x in rows),accounting
 yield 'metadata.json',json.dumps(dict(report_sha256=digest(b),actual_terminal=accounting)).encode()
 yield 'report.json',b
 yield 'initial.pt',(RUN/'initial.pt').read_bytes()
 for row in r['steps']:
  for rel in row['source']:yield rel,(ROOT/rel).read_bytes()
  for key in ('arrays_path','checkpoint'):yield row[key],(ROOT/row[key]).read_bytes()
def incoming():
 with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as archive:
  for member in archive:
   assert member.isfile() and 0<=member.size<=512*1024*1024 and not member.name.startswith('/') and '..' not in Path(member.name).parts
   b=archive.extractfile(member).read();assert len(b)==member.size;yield member.name,b
def exported():
 with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
  for name,b in blobs():
   info=tarfile.TarInfo(name);info.size=len(b);archive.addfile(info,io.BytesIO(b))
def audit(iterator,output):
 it=iter(iterator)
 def get(name):
  actual,b=next(it);assert actual==name,(actual,name);return b
 metadata=json.loads(get('metadata.json'));b=get('report.json');assert digest(b)==metadata['report_sha256'];r=json.loads(b)
 result=dict(state='failed',checked_steps=0,complete_saved_values=0,gradient_values=0,optimizer_parameters_checked=0,max_errors=dict(input=0.,loss=0.,norm=0.,clip=0.,moment=0.,parameter=0.),formal_training_closed=False,AP_measured=False)
 try:
  lock=ROOT/'experiments/original-paper-scenarios/ecsic-KITTI-GPU-code-001/inputs.json';locked=json.loads(lock.read_text());assert sha(lock)==r['input_sha256']
  assert r['sources_before']==r['sources_after']==locked['files'] and r['base_freeze_before']==r['base_freeze_after']==locked['base_freeze']
  assert r['GPU'].find('RTX PRO 6000')>=0 and r['capability']==[12,0] and len(r['steps'])==2 and not r['AP_measured']
  assert r['peak_reserved_bytes']+2*1024**3<=r['initial_free_bytes']
  b=get('initial.pt');assert digest(b)==r['initial_checkpoint_sha256'];previous=safe(b)
  public=torch.load(ROOT/'assets/ecsic-cs001-001/model.pt',map_location='cpu',weights_only=True)
  assert sha(ROOT/'assets/ecsic-cs001-001/model.pt')==locked['public_weight_sha256']
  assert states(previous['model'])==states(public)==r['initial_states'] and len(public)==225;del public
  assert previous['optimizer']['state']=={}
  names=r['parameter_names'];assert len(names)==len(set(names));assert sum(previous['model'][k].numel() for k in names)==r['parameter_count']
  for index,row in enumerate(r['steps']):
   assert row['step']==index and row['frame']==('000000','000003')[index] and row['lambda_RD']==.01
   sources=[]
   for folder in ('image_2','image_3'):
    rel=f'data/kitti/training/{folder}/{row["frame"]}.png';b=get(rel);item=row['source'][rel]
    assert len(b)==item['bytes'] and digest(b)==item['sha256'];sources.append(np.asarray(Image.open(io.BytesIO(b)).convert('RGB')).copy())
   b=get(row['arrays_path']);assert digest(b)==row['arrays_sha256']
   with np.load(io.BytesIO(b),allow_pickle=False) as z:a={k:z[k] for k in z.files}
   assert set(a)==set(row['arrays'])
   for k,v in a.items():assert np.isfinite(v).all() and descriptor(v)==row['arrays'][k],k
   result['complete_saved_values']+=sum(v.size for v in a.values())
   assert sources[0].shape==sources[1].shape;h,w=sources[0].shape[:2];assert row['original_hw']==[h,w]
   ph,pw=(-h)%32,(-w)%32;assert row['padded_hw']==[h+ph,w+pw]
   for view,source in zip(('left','right'),sources):
    expected=np.pad(source.astype(np.float64)/255.,((0,ph),(0,pw),(0,0)),mode='edge').transpose(2,0,1)[None]
    result['max_errors']['input']=max(result['max_errors']['input'],near(a['input_'+view],expected,4*EPS*(1+np.abs(expected)),view))
   mse=sum(float(np.square(a['input_'+view].astype(np.float64)-a['pred_'+view].astype(np.float64)).mean())*255**2 for view in ('left','right'))/2
   bpp=sum(float(a['rate_'+k].astype(np.float64).mean()) for k in ('left_y','left_z','right_y','right_z'))/((h+ph)*(w+pw)*2)
   expected=np.asarray([mse,bpp,(bpp+.01*mse)/1.01]);assert a['losses'].shape==(3,) and (a['losses']>=0).all()
   result['max_errors']['loss']=max(result['max_errors']['loss'],near(a['losses'],expected,64*EPS*(1+np.abs(expected)),'RD loss'))
   assert states(previous['model'])==row['states_before']
   b=get(row['checkpoint']);assert digest(b)==row['checkpoint_sha256'];current=safe(b);assert states(current['model'])==row['states_after']
   group=current['optimizer']['param_groups'];assert len(group)==1;g=group[0]
   assert g['lr']==1e-4 and tuple(g['betas'])==(.9,.999) and g['eps']==1e-8 and g['weight_decay']==0
   assert not any(g.get(k,False) for k in ('amsgrad','maximize','capturable','differentiable')) and g.get('fused') in (None,False)
   assert len(g['params'])==len(names) and set(current['optimizer']['state'])==set(g['params'])
   for prefix in ('gradient_','clipped_gradient_'):assert {k for k in a if k.startswith(prefix)}=={prefix+k for k in names}
   norm=math.sqrt(sum(float(np.square(a['gradient_'+k].astype(np.float64)).sum()) for k in names))
   result['max_errors']['norm']=max(result['max_errors']['norm'],near(np.asarray(row['gradient_norm']),np.asarray(norm),64*EPS*(1+norm),'gradient norm'))
   coefficient=min(1.,2.5/(row['gradient_norm']+1e-6));t=index+1
   for paramid,name in zip(g['params'],names):
    raw=a['gradient_'+name].astype(np.float64);grad=a['clipped_gradient_'+name].astype(np.float64)
    result['max_errors']['clip']=max(result['max_errors']['clip'],near(grad,raw*coefficient,8*EPS*np.abs(raw*coefficient)+EPS,'clipping'))
    moment=current['optimizer']['state'][paramid];old=previous['optimizer']['state'].get(paramid)
    assert set(moment)=={'step','exp_avg','exp_avg_sq'} and float(moment['step'])==t
    pm=np.zeros_like(grad) if old is None else old['exp_avg'].numpy().astype(np.float64)
    pv=np.zeros_like(grad) if old is None else old['exp_avg_sq'].numpy().astype(np.float64)
    if old is not None:assert float(old['step'])==t-1
    m=.9*pm+.1*grad;v=.999*pv+.001*grad*grad
    for key,ref,bound in [('exp_avg',m,128*EPS*(.9*np.abs(pm)+.1*np.abs(grad)+np.finfo(np.float32).tiny)),('exp_avg_sq',v,128*EPS*(.999*np.abs(pv)+.001*grad*grad+np.finfo(np.float32).tiny))]:
     assert torch.isfinite(moment[key]).all();result['max_errors']['moment']=max(result['max_errors']['moment'],near(moment[key].numpy(),ref,bound,key))
    parameter=previous['model'][name].numpy().astype(np.float64);delta=1e-4/(1-.9**t)*m/(np.sqrt(v/(1-.999**t))+1e-8);ref=parameter-delta
    result['max_errors']['parameter']=max(result['max_errors']['parameter'],near(current['model'][name].numpy(),ref,128*EPS*(1+np.abs(parameter)+np.abs(delta)+np.abs(ref)),name))
    result['gradient_values']+=raw.size+grad.size;result['optimizer_parameters_checked']+=raw.size
   assert all(torch.isfinite(v).all() for v in current['model'].values())
   previous=current;result['checked_steps']+=1;del a
  try:next(it)
  except StopIteration:pass
  else:raise AssertionError('Unexpected suffix')
  result['state']='passed_complete_two_native_RGB_RD_Adam_steps'
 except BaseException:result['traceback']=traceback.format_exc();raise
 finally:
  result.update(metadata=metadata,verifier_sha256=sha(__file__),protocol_sha256=sha(ROOT/'experiments/original-paper-scenarios/ecsic-KITTI-GPU-protocol-001.md'),checked_unix=time.time(),limitation='Complete saved tensors and independent RGB/RD/Adam arithmetic; no independent CUDA gradient, actual entropy-bitstream rate, full KITTI training or AP claim.')
  output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:result[k] for k in ('state','checked_steps','complete_saved_values','gradient_values')}))
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=('native','export','stream'),required=True);p.add_argument('--output',type=Path);a=p.parse_args();torch.set_num_threads(2)
 if a.mode=='export':exported();return
 assert a.output and not a.output.exists();audit(incoming() if a.mode=='stream' else blobs(),a.output)
if __name__=='__main__':main()
