"""Paired preparation with the unchanged verified symbol-leaf receiver path."""
import hashlib
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'code'))
from native_transport import RiskChannel,array_identity,global_rng_identity
from pair_noise import PairNoise


class AntitheticChannel(RiskChannel):
    def __init__(self,link,output_directory,layouts):
        super().__init__(link,output_directory,layouts)
        self.pairs=PairNoise(self.group_ids);self.rng=self.pairs.rng

    def prepare(self,frame,mode,group=None,pair=None,sign=None):
        if self.closed or self.pending is not None:raise ValueError('unfinished/closed channel')
        stem=f'{frame}-{mode}'+(f'-g{group:02d}-p{pair:02d}-s{sign:+d}' if mode=='group' else '')
        path=self.directory/(stem+'.npy')
        if path.exists():raise RuntimeError('preserve previous actual noise')
        noise,record=self.pairs.sample(frame,mode,group,pair,sign)
        self.prepared+=1
        with path.open('xb') as f:np.save(f,noise,allow_pickle=False)
        self.pending=dict(attempt=self.prepared,frame_id=frame,mode=mode,group=group,draw=pair,pair=pair,sign=sign,
                          noise_path=str(path.resolve()),noise_array_sha256=array_identity(noise),
                          noise_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),noise_energy=float(np.square(noise.astype(np.float64)).sum()),
                          PCG64=record,global_rng_before=global_rng_identity(),consumed=False)
        self.noise=noise;self.leaf=None;self.last=None
