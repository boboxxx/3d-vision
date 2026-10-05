#!/usr/bin/env python3
"""One real preprocessed frame, all fixed pooling conditions; engineering only."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import numpy as np
import torch
import torch.distributed as dist
ROOT=Path(__file__).resolve().parents[1];LIGA=ROOT/'third_party/LIGA-Stereo'
sys.path[:0]=[str(ROOT/'src'),str(LIGA),str(ROOT/'third_party/mmdetection_kitti')]
from geocomm.pooling_diagnostic import CONDITIONS,PoolingDiagnostic,state_hashes
from geocomm.evidence import sha256,source_identity
from diagnose_pooling import frozen_load


def main():
    p=argparse.ArgumentParser()
    for name in ('config','checkpoint','output'): p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    for key,value in vars(args).items(): setattr(args,key,value.resolve())
    if args.output.exists(): p.error('preserve prior engineering evidence')
    random_seed=17;torch.manual_seed(random_seed);np.random.seed(random_seed);torch.cuda.manual_seed_all(random_seed)
    free_before,total=torch.cuda.mem_get_info()
    if free_before<10*1024**3: raise RuntimeError('insufficient GPU margin for coexistence engineering')
    os.chdir(LIGA)
    from easydict import EasyDict
    from liga.config import cfg_from_yaml_file
    from liga.datasets import build_dataloader
    from liga.models import build_network
    from liga.utils.common_utils import create_logger
    cfg=cfg_from_yaml_file(str(args.config),EasyDict())
    if cfg.MODEL.BACKBONE_3D.SEMANTIC_LINK.enabled or not cfg.MODEL.BACKBONE_3D.STUDENT_ENCODER.allow_uncompressed_diagnostic:
        raise RuntimeError('explicit uncompressed diagnostic required')
    dataset,loader,_=build_dataloader(cfg.DATA_CONFIG,cfg.CLASS_NAMES,batch_size=1,dist=False,workers=0,training=False,logger=create_logger())
    batch=next(iter(loader));sensors={key:batch[key] for key in ('batch_size','left_img','right_img','calib','image_shape','frame_id')}
    for key in ('left_img','right_img'): sensors[key]=torch.as_tensor(sensors[key],device='cuda',dtype=torch.float32)
    with tempfile.TemporaryDirectory(prefix='pool-probe-dist-') as directory:
        dist.init_process_group('nccl',init_method='file://'+directory+'/rank',rank=0,world_size=1)
        try:
            model=build_network(cfg.MODEL,len(cfg.CLASS_NAMES),dataset).cuda().eval();loading=frozen_load(model,args.checkpoint)
            # Cover the same native DDP wrapping used by complete evaluation.
            wrapped=torch.nn.parallel.DistributedDataParallel(model,device_ids=[0],broadcast_buffers=False)
            records=[]
            for condition in CONDITIONS:
                rows=[];observer=PoolingDiagnostic(model,condition,rows.append);torch.cuda.reset_peak_memory_stats()
                try:
                    with torch.no_grad(): predictions,_=wrapped(dict(sensors))
                    torch.cuda.synchronize()
                    if state_hashes(model)!=loading['state_hashes']: raise RuntimeError('engineering changed frozen model')
                    if observer.calls!=dict(frames=1,student=2,raw_cost=1,forbidden=0): raise RuntimeError('engineering module sequence differs')
                    records.append(dict(condition=condition,rows=rows,calls=observer.calls,
                        prediction_count=len(predictions[0]['pred_boxes']),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved()))
                    del predictions
                finally: observer.close()
            peak=max(row['peak_reserved_bytes'] for row in records)
            if peak+2*1024**3>free_before: raise RuntimeError('engineering lacks2GiB measured coexistence margin')
            result=dict(state='passed',scope='one real frame engineering only; no full-fold AP or communication claim',
                seed=17,native_DDP_wrapper=True,distributed_backend='nccl',frame_id=str(sensors['frame_id'][0]),native_preprocessed_shape=list(sensors['left_img'].shape),
                gpu_free_before_bytes=free_before,gpu_total_bytes=total,conditions=records,
                checkpoint_loading=loading,checkpoint_sha256=sha256(args.checkpoint),config_sha256=sha256(args.config),
                project_sources=source_identity(ROOT,['src','scripts','configs','pyproject.toml']),
                torch=torch.__version__,gpu=torch.cuda.get_device_name(0),required_states_identical=519)
            args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps({k:result[k] for k in ['state','frame_id','native_preprocessed_shape','gpu_free_before_bytes','required_states_identical']}))
        finally: dist.destroy_process_group()


if __name__=='__main__': main()
