"""Fixed symbol-leaf interventions; no model parameters, private decoder inputs or policy."""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from proxy import groups, group_noise


def array_identity(value):
    if isinstance(value,torch.Tensor):value=value.detach().cpu().numpy()
    value=np.ascontiguousarray(value)
    payload=json.dumps({'dtype':value.dtype.str,'shape':list(value.shape)},sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(payload+b'\0'+value.tobytes()).hexdigest()


def global_rng_identity():
    state=np.random.get_state()
    numpy_digest=hashlib.sha256(state[0].encode()+state[1].tobytes()+json.dumps(state[2:]).encode()).hexdigest()
    return {'numpy':numpy_digest,'torch_CPU':array_identity(torch.get_rng_state()),
            'torch_CUDA':[array_identity(s) for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


class RiskChannel:
    def __init__(self,link,output_directory,layouts,seed=2801):
        if link.channel.kind!='identity' or link.training or seed!=2801:
            raise ValueError('fixed eval identity architecture and PCG64 seed2801 required')
        self.link=link;self.channel=link.channel;self.original=self.channel.forward
        self.layouts=layouts;self.group_ids=groups(layouts);self.rng=np.random.Generator(np.random.PCG64(seed))
        self.directory=Path(output_directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.pending=None;self.leaf=None;self.calls=0;self.prepared=0;self.reference_tx=None;self.last=None;self.closed=False
        self.handle=self.channel.register_forward_pre_hook(self.leaf_input,prepend=True)
        self.channel.forward=self.forward

    def begin_frame(self):
        if self.pending is not None:raise RuntimeError('unfinished previous channel attempt')
        self.reference_tx=None

    def prepare(self,frame,mode,group=None,draw=None):
        if self.closed or self.pending is not None or mode not in ('clean','group','full_awgn'):
            raise ValueError('unprepared, overlapping or unknown intervention')
        before=copy.deepcopy(self.rng.bit_generator.state);count=len(self.group_ids)
        if mode=='group':
            if type(draw) is not int or not 0<=draw<4:raise ValueError('fixed four draw indices')
            noise,record=group_noise(self.group_ids,group,rng=self.rng)
        elif mode=='full_awgn':
            if group is not None or draw is not None:raise ValueError('full noise has no selected cell')
            noise=(self.rng.standard_normal((count,2))*math.sqrt(.1/2)).astype(np.float32)
            record=dict(before=before,after=copy.deepcopy(self.rng.bit_generator.state),draw_shape=[count,2],selected_symbols=count,attempted_complex_uses=count,local_N0=.1)
        else:
            if group is not None or draw is not None:raise ValueError('clean noise has no selected cell')
            noise=np.zeros((count,2),np.float32)
            record=dict(before=before,after=copy.deepcopy(before),draw_shape=[0,2],selected_symbols=0,attempted_complex_uses=count,local_N0=0.)
        self.prepared+=1
        stem=f'{frame}-{mode}'+(f'-g{group:02d}-r{draw}' if mode=='group' else '')
        path=self.directory/(stem+'.npy')
        if path.exists():raise RuntimeError('preserve previous noise array')
        np.save(path,noise,allow_pickle=False)
        self.pending=dict(attempt=self.prepared,frame_id=frame,mode=mode,group=group,draw=draw,
                          noise_path=str(path.resolve()),noise_array_sha256=array_identity(noise),
                          noise_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),noise_energy=float(np.square(noise.astype(np.float64)).sum()),
                          PCG64=record,global_rng_before=global_rng_identity(),consumed=False)
        self.noise=noise;self.leaf=None;self.last=None

    def leaf_input(self,module,inputs):
        if self.pending is None or self.pending['consumed'] or len(inputs)!=2:
            raise RuntimeError('channel outside one prepared attempt')
        symbols,snr=inputs
        if symbols.shape!=(1,len(self.group_ids),2) or symbols.dtype!=torch.float32 or not torch.isfinite(symbols).all() or float(snr)!=10.:
            raise ValueError('fixed FP32 physical symbol layout/SNR')
        if symbols.requires_grad:raise RuntimeError('all transmitter parameters must stay frozen')
        identity=array_identity(symbols)
        if self.reference_tx is None:
            if self.pending['mode']!='clean':raise RuntimeError('first frame attempt must be clean')
            self.reference_tx=identity
        if identity!=self.reference_tx:raise RuntimeError('forward symbols changed between interventions')
        self.pending['transmitted_symbols_sha256']=identity
        self.leaf=symbols.detach().requires_grad_(True)
        return self.leaf,snr

    def forward(self,symbols,snr_db,generator=None):
        if self.pending is None or self.pending['consumed'] or symbols is not self.leaf or generator is not None:
            raise RuntimeError('unexpected channel invocation or external generator')
        received,account=self.original(symbols,snr_db)
        if received is not symbols:raise RuntimeError('identity parent unexpectedly transformed symbols')
        # Preserve identity bits, including negative zero, when no draw exists.
        noisy=symbols if self.pending['mode']=='clean' else symbols+torch.as_tensor(self.noise,device=symbols.device)[None]
        self.pending['received_symbols_sha256']=array_identity(noisy)
        self.pending['global_rng_after']=global_rng_identity()
        if self.pending['global_rng_after']!=self.pending['global_rng_before']:
            raise RuntimeError('channel changed global model/data RNG')
        label={'clean':'identity','group':'structured_group_CN01','full_awgn':'awgn'}[self.pending['mode']]
        account=dict(account,channel=label,intervention_group=self.pending['group'])
        self.pending['consumed']=True;self.calls+=1;self.last=copy.deepcopy(self.pending)
        return noisy,account

    def finish(self):
        if self.pending is None or not self.pending['consumed'] or self.calls!=self.prepared:
            raise RuntimeError('missing/repeated channel call')
        result=self.last;self.pending=None;self.noise=None;self.leaf=None
        return result

    def close(self):
        if not self.closed:
            self.channel.forward=self.original;self.handle.remove();self.closed=True


def cell_map(h,w):
    rows=np.minimum(((np.arange(h)+.5)*4/h).astype(int),3)
    cols=np.minimum(((np.arange(w)+.5)*8/w).astype(int),7)
    return rows[:,None]*8+cols[None]


def aggregate_geometry(result,symbols,gradient,group_ids):
    probability=result['left']['probability']
    ids=cell_map(*probability.shape[1:])
    output=[]
    for group in range(32):
        selected=group_ids==group
        item=dict(group=group,symbol_count=int(selected.sum()),code_energy_mean=float(np.square(symbols[selected].astype(np.float64)).sum(1).mean()),
                  squared_gradient_sum=float(np.square(gradient[selected].astype(np.float64)).sum()))
        item['first_order_loss_variance_proxy']=.05*item['squared_gradient_sum']
        valid=[(ids==group)&result[view]['valid'] for view in ('left','right')]
        item['valid_count']=sum(int(mask.sum()) for mask in valid)
        item['candidate_count']=2*int((ids==group).sum())
        for name in ('entropy','depth_mean','depth_variance','disparity_mean','disparity_variance','support_count'):
            values=np.concatenate([result[view][name][mask] for view,mask in zip(('left','right'),valid)])
            item[name]=float(values.mean()) if len(values) else None
        output.append(item)
    return output
