"""All eight author MAT models against independent literal channel correlations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
from scipy.ndimage import correlate
import torch

from core import AuthorCore,load,fingerprint

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
UPSTREAM=ROOT/'third_party/SRCNN-official-001'
PROTOCOL_SHA='f65369ab9e2f344e95975945b2847150aa95b99f5ad91a5af6c977f283f8986a'
ARCHIVES={'SRCNN_v1.zip':'bfa68ca613c1326a59e0c34353205a254ab2b67e34df7f04e28eef567980af30',
          'SRCNN_train.zip':'001146419f7acfb12a3e7929c8acd5de88a08d687d6881085f81321ad6982b1a'}
MODELS=['9-1-5(91 images)/x2.mat','9-1-5(91 images)/x3.mat','9-1-5(91 images)/x4.mat',
        '9-1-5(ImageNet)/x3.mat','9-3-5(ImageNet)/x3.mat','9-5-5(ImageNet)/x2.mat',
        '9-5-5(ImageNet)/x3.mat','9-5-5(ImageNet)/x4.mat']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def reference(data,image):
    # Interpret original flattened MAT slices directly, not Torch-translated weights.
    a=np.empty((*image.shape,64),dtype=np.float64)
    for i in range(64):
        kernel=np.reshape(data['weights_conv1'][:,i],(9,9),order='F')
        a[:,:,i]=np.maximum(correlate(image,kernel,mode='nearest')+data['biases_conv1'][i,0],0)
    k=int(np.sqrt(data['weights_conv2'].shape[1]));b=np.zeros((*image.shape,32),dtype=np.float64)
    for i in range(32):
        for j in range(64):
            kernel=np.reshape(data['weights_conv2'][j,:,i],(k,k),order='F')
            b[:,:,i]+=correlate(a[:,:,j],kernel,mode='nearest')
        b[:,:,i]=np.maximum(b[:,:,i]+data['biases_conv2'][i,0],0)
    result=np.zeros(image.shape,dtype=np.float64)
    for i in range(32):
        kernel=np.reshape(data['weights_conv3'][i,:],(5,5),order='F')
        result+=correlate(b[:,:,i],kernel,mode='nearest')
    return result+data['biases_conv3'][0,0]


def barrier(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes)):return
    name=os.fsdecode(args[0]);mode,flags=args[1:3]
    reading=(isinstance(mode,str) and 'r' in mode) or (mode is None and flags & os.O_ACCMODE==os.O_RDONLY)
    if reading:assert '/data/kitti/' not in name and not name.endswith(('.png','.pth','.pt','.npz','.pkl')), 'foreign dataset/model input forbidden'


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();assert not args.output.exists()
    protocol=HERE.parent/'srcnn-author-core-protocol-001.md';assert sha(protocol)==PROTOCOL_SHA
    sources={str(path.relative_to(ROOT)):sha(path) for path in sorted(HERE.glob('*.py'))};assert len(sources)==2
    for name,digest in ARCHIVES.items():assert sha(UPSTREAM/name)==digest
    base=UPSTREAM/'SRCNN_v1/SRCNN'
    assert {str(path.relative_to(base/'model')) for path in (base/'model').glob('*/*.mat')}==set(MODELS)
    upstream_sources={str(path.relative_to(ROOT)):sha(path) for path in base.glob('*.m')}
    for path in (UPSTREAM/'SRCNN_train/SRCNN').iterdir():
        if path.suffix in ('.m','.txt','.prototxt'):upstream_sources[str(path.relative_to(ROOT))]=sha(path)
    zero=np.zeros((16,20),np.float64);impulse=zero.copy();impulse[7,11]=1
    inputs=dict(zero=zero,impulse=impulse,uniform=np.random.Generator(np.random.PCG64(20261004)).uniform(size=(16,20)))
    torch.set_num_threads(2);sys.addaudithook(barrier);records=[];weights={};values=0
    with torch.no_grad():
        for name in MODELS:
            path=base/'model'/name;data=load(path);model=AuthorCore(data).eval()
            initial={k:fingerprint(v.numpy()) for k,v in model.state_dict().items()}
            weights[name]=dict(file_sha256=sha(path),arrays={k:fingerprint(v) for k,v in data.items()},readonly_tensors=initial)
            for label,image in inputs.items():
                actual=model(torch.tensor(image).reshape(1,1,16,20)).numpy()[0,0]
                expected=reference(data,image);error=float(np.abs(actual-expected).max())
                assert actual.shape==expected.shape==(16,20) and np.isfinite(actual).all() and error<=1e-10
                assert {k:fingerprint(v.numpy()) for k,v in model.state_dict().items()}==initial
                records.append(dict(model=name,input=label,maximum_absolute_error=error,torch_output=fingerprint(actual),reference_output=fingerprint(expected)))
                values+=actual.size
            assert sha(path)==weights[name]['file_sha256']
    assert len(records)==24 and values==7680 and all(sha(ROOT/p)==h for p,h in {**sources,**upstream_sources}.items())
    assert all(sha(UPSTREAM/name)==h for name,h in ARCHIVES.items())
    result=dict(state='passed_all8_author_models24_double_input_mathematical_core_comparisons',checked_unix=time.time(),
                sources=sources,protocol_sha256=PROTOCOL_SHA,archives=ARCHIVES,upstream_sources=upstream_sources,weights=weights,
                records=records,output_values=values,input_identities={k:fingerprint(v) for k,v in inputs.items()},
                runtime=dict(Python=platform.python_version(),NumPy=np.__version__,SciPy=scipy.__version__,PyTorch=torch.__version__),
                no_KITTI_GT_model_update_or_detector=True,
                limitation='No MATLAB runtime or demo mixed-precision/preprocessing replay, no10/30/50compression interface or AP')
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(state=result['state'],comparisons=24,maximum_absolute_error=max(r['maximum_absolute_error'] for r in records))),flush=True)


if __name__=='__main__':main()
