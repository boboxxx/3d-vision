"""Registry-bound official evaluation graph; decode receives no source inputs."""
import hashlib,importlib,json,sys,types
from pathlib import Path
import numpy as np
import torch
import codec as c
from registry import sha,states,ROOT,LAMBDAS,CONFIG,CONFIG_SHA,DATASET_SHA

def load(registry_path,registry_sha,candidate,device='cpu'):
 assert 0<=candidate<6 and sha(registry_path)==registry_sha
 registry=json.loads(Path(registry_path).read_text());assert registry['schema']=='ecsic-final-KITTI-registry-v1' and registry['state']=='closed_all_six_final_KITTI_models_with_native_local_full_training_admission'
 assert len(registry['entries'])==6 and [(e['candidate'],e['lambda_RD'],e['seed'],e['epoch'],e['updates']) for e in registry['entries']]==[(i,v,17,10,37120) for i,v in enumerate(LAMBDAS)]
 assert registry['config_sha256']==CONFIG_SHA==sha(ROOT/CONFIG) and registry['dataset_manifest_sha256']==DATASET_SHA
 entry=registry['entries'][candidate];path=(ROOT/entry['model_path']).resolve();assert path.is_relative_to(ROOT) and sha(path)==entry['model_sha256'] and path.stat().st_size==entry['model_bytes']
 source=ROOT/'data/provenance/ECSIC-official-source-audit-001.json';assert sha(source)==registry['official_source_audit_sha256'];official=json.loads(source.read_text());assert official['commit']=='696f4ae4f250bb1fc750ae9ec23e9c98c2c7e6db'
 for name,digest in official['files'].items():assert sha(ROOT/'third_party/ECSIC-official-001'/name)==digest
 sys.path.insert(0,str(ROOT/'data/engineering/ecsic-import-overlay-001'))
 def disabled(*a,**k):raise RuntimeError('External logger forbidden')
 logger=types.ModuleType('wandb');logger.__file__='disabled-import-only-wandb'
 for name in ['init','log','watch','finish']:setattr(logger,name,disabled)
 logger.util=types.SimpleNamespace(generate_id=disabled);sys.modules['wandb']=logger
 package=types.ModuleType('ecsic');package.__path__=[str(ROOT/'third_party/ECSIC-official-001/ecsic')];sys.modules['ecsic']=package
 models=importlib.import_module('ecsic.models');utils=importlib.import_module('ecsic.utils');cfg=json.loads((ROOT/CONFIG).read_text())['model'];model=getattr(models,cfg['name'])(**cfg['kwargs'])
 weights=torch.load(path,map_location='cpu',weights_only=True);assert len(weights)==225 and weights.keys()==model.state_dict().keys() and states(weights)==entry['states'] and all(torch.isfinite(v).all() for v in weights.values());model.load_state_dict(weights,strict=True);assert states(model.state_dict())==entry['states']
 assert len(list(model.parameters()))==216 and sum(p.numel() for p in model.parameters())==31640742
 for k in entry['shared_aliases']:assert torch.equal(weights[k],weights[k.replace('.layer_right.','.layer_left.')])
 model=model.to(device).eval().requires_grad_(False);return model,utils,entry

def array(v):return v.detach().cpu().contiguous().numpy()

def hooks(model,receiver=False):
 counts=dict(E=0,HE=0,HD=0,D=0);handles=[]
 for name in counts:
  def callback(module,inputs,name=name):
   assert not(receiver and name in ['E','HE']),'Receiver invoked source encoder';counts[name]+=1
  handles.append(getattr(model,name).register_forward_pre_hook(callback))
 return counts,handles

@torch.inference_mode()
def encode(model,utils,left,right,model_sha):
 assert left.shape==right.shape and left.ndim==4 and left.shape[:2]==(1,3) and left.dtype==right.dtype==torch.float32 and torch.isfinite(left).all() and torch.isfinite(right).all()
 h,w=map(int,left.shape[-2:]);ph,pw=(h+31)//32*32,(w+31)//32*32
 padded=[torch.nn.functional.pad(v,(0,pw-w,0,ph-h),mode='replicate') for v in [left,right]];pos=utils.get_positional_fourier_encoding(ph,pw).unsqueeze(0).to(left.device);ref=model(*padded,pos=pos)
 yl,yr,_,_=model.E(*padded,pos=pos,return_attn=True);zl,zr,_,_=model.HE(yl,yr,pos=pos,return_attn=True);original=dict(z_left=zl,z_right=zr,y_left=yl,y_right=yr);streams={};arrays={};bins={};cdfs,_=c.tables()
 for name in c.ORDER:
  level,side=name.split('_');lat=ref.latents[side];loc,scale=lat[level+'_loc'],lat[level+'_scale'];q=torch.round(original[name]-loc)
  assert torch.isfinite(q).all() and q.abs().max()<2**24 and torch.equal(q+loc,lat[level+'_hat_dec']) and torch.equal(torch.round(lat[level+'_hat_dec']-loc),q)
  residual=array(q.to(torch.int32));assert residual.shape==c.shape_of(c.ORDER.index(name),[ph,pw]);streams[name]=c.encode_stream(residual,array(scale),cdfs);bins[name]=c.scale_ids(array(scale),residual.shape)
  arrays.update({name+'_symbols':residual,name+'_loc':array(loc),name+'_scale':array(scale),name+'_hat':array(lat[level+'_hat_dec'])})
 arrays.update(pred_left=array(ref.pred.left),pred_right=array(ref.pred.right));assert len(arrays)==18 and all(np.isfinite(v).all() for v in arrays.values())
 return c.pack([h,w],[ph,pw],streams,model_sha),arrays,bins

@torch.inference_mode()
def decode(model,utils,parsed):
 h,w=parsed['padded_hw'];device=next(model.parameters()).device;pos=utils.get_positional_fourier_encoding(h,w).unsqueeze(0).to(device);arrays={};bins={};cdfs,_=c.tables()
 def pull(index,loc,scale):
  name=c.ORDER[index];shape=c.shape_of(index,[h,w]);sc=array(scale);q=c.decode_stream(*parsed['streams'][name],sc,shape,cdfs);value=torch.from_numpy(q).to(device).float()+loc;bins[name]=c.scale_ids(sc,shape)
  arrays.update({name+'_symbols':q,name+'_loc':array(loc),name+'_scale':sc,name+'_hat':array(value)});return value
 zl=pull(0,model.zl_loc,model.zl_scale);zr=pull(1,*model.zr_entropy(zl));left,right,_,_=model.HD(zl,zr,pos=pos,return_attn=True)
 yl=pull(2,*model.hd_out_left(left).chunk(2,1));yr=pull(3,*model.yr_entropy(yl,model.hd_out_right(right),pos=pos)[:2]);pl,pr,_,_=model.D(yl,yr,pos=pos,return_attn=True);arrays.update(pred_left=array(pl),pred_right=array(pr))
 assert len(arrays)==18 and all(np.isfinite(v).all() for v in arrays.values());return arrays,bins
