"""Native sensor-only paired RGB from the fixed, independently audited ROI cache."""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class StereoRGB(Dataset):
    def __init__(self, root, records, audit, fold, split):
        self.root=Path(root);self.records_path=Path(records)
        self.audit=json.loads(Path(audit).read_text());self.fold=json.loads(Path(fold).read_text())
        self.rows=[json.loads(line) for line in self.records_path.read_text().splitlines()]
        if (self.audit['state']!='passed' or self.audit['records_sha256']!=digest(records)
                or self.audit['fold_sha256']!=digest(fold)
                or self.audit['split']!=split or [row['frame_id'] for row in self.rows]!=self.fold['folds'][split]['ids']
                or len(self.rows)!=self.audit['frames']):
            raise ValueError('audited ROI/fold identity differs')
        self.split=split

    def __len__(self): return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index];images=[];boxes=[]
        if [view['camera'] for view in row['views']]!=['image_2','image_3']:
            raise ValueError('native stereo order differs')
        for view in row['views']:
            raw=(self.root/'training'/view['camera']/(row['frame_id']+'.png')).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=view['image_sha256']:
                raise ValueError('sensor image identity differs: '+row['frame_id'])
            with Image.open(io.BytesIO(raw)) as image:
                if image.mode!='RGB' or [image.height,image.width]!=view['shape']:
                    raise ValueError('native RGB/shape differs')
                array=np.asarray(image).copy()
            images.append(torch.from_numpy(array).permute(2,0,1).unsqueeze(0).float().contiguous()/255.)
            boxes.append(view['boxes'])
        if images[0].shape!=images[1].shape: raise ValueError('native paired shapes differ')
        return dict(frame_id=row['frame_id'],left=images[0],right=images[1],boxes=boxes)


def single_frame(rows):
    if len(rows)!=1: raise ValueError('native variable-shape frames require batch1')
    return rows[0]
