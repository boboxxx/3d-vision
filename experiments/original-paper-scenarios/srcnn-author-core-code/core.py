"""Exact filter layout and replicate-boundary mathematical SRCNN.m core."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
from torch.nn import functional as F


KEYS=('weights_conv1','biases_conv1','weights_conv2','biases_conv2','weights_conv3','biases_conv3')


def load(path):
    data=loadmat(path);assert {k for k in data if not k.startswith('__')}==set(KEYS)
    assert all(np.issubdtype(data[k].dtype,np.floating) and np.isfinite(data[k]).all() for k in KEYS)
    assert data['weights_conv1'].shape==(81,64) and data['biases_conv1'].shape==(64,1)
    assert data['weights_conv2'].shape[0]==64 and data['weights_conv2'].shape[2]==32
    assert data['weights_conv2'].shape[1] in (1,9,25)
    assert data['weights_conv3'].shape==(32,25) and data['biases_conv2'].shape==(32,1) and data['biases_conv3'].shape==(1,1)
    return {k:data[k] for k in KEYS}


def fingerprint(a):
    a=np.ascontiguousarray(a);schema=dict(dtype=a.dtype.str,shape=list(a.shape))
    return dict(**schema,sha256=hashlib.sha256(json.dumps(schema,sort_keys=True).encode()+b'\0'+a.tobytes()).hexdigest())


class AuthorCore(nn.Module):
    def __init__(self,data):
        super().__init__();k=int(np.sqrt(data['weights_conv2'].shape[1]));self.kernel_sizes=(9,k,5)
        w1=np.stack([data['weights_conv1'][:,i].reshape(9,9,order='F') for i in range(64)])[:,None]
        w2=np.stack([np.stack([data['weights_conv2'][j,:,i].reshape(k,k,order='F') for j in range(64)]) for i in range(32)])
        w3=np.stack([data['weights_conv3'][i,:].reshape(5,5,order='F') for i in range(32)])[None]
        for index,w in enumerate((w1,w2,w3),1):
            self.register_buffer('weight'+str(index),torch.tensor(w,dtype=torch.float64))
            self.register_buffer('bias'+str(index),torch.tensor(data['biases_conv'+str(index)].ravel(),dtype=torch.float64))
        assert len(self.state_dict())==6 and not list(self.parameters())

    def forward(self,value):
        assert value.dtype==torch.float64 and value.ndim==4 and value.shape[:2]==(1,1) and value.device.type=='cpu'
        assert torch.isfinite(value).all() and not torch.is_grad_enabled()
        for index,k in enumerate(self.kernel_sizes,1):
            value=F.conv2d(F.pad(value,(k//2,)*4,mode='replicate'),getattr(self,'weight'+str(index)),getattr(self,'bias'+str(index)))
            if index<3:value=F.relu(value)
        return value
