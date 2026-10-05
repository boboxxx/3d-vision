#!/usr/bin/env python3
"""One fixed exploratory codec epoch; full receiver and native losses retained."""
import argparse
from pathlib import Path
import yaml


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--allocation',choices=['uniform','geometry','task','geometry_task'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('preserve existing tuning config')
    cfg=yaml.safe_load(args.base.read_text())
    assert cfg['MODEL']['BACKBONE_3D']['SEMANTIC_LINK']['task_sensitivity']
    cfg['MODEL']['BACKBONE_3D']['SEMANTIC_LINK']['allocation']=args.allocation
    cfg['DATA_CONFIG'].update(DATA_SPLIT=dict(train='geocomm_tune_train',test='geocomm_tune_holdout'),
        INFO_PATH=dict(train=['kitti_infos_geocomm_tune_train.pkl'],test=['kitti_infos_geocomm_tune_holdout.pkl']))
    cfg['OPTIMIZATION'].update(NUM_EPOCHS=1,BATCH_SIZE_PER_GPU=1,OPTIMIZER='adamw',LR=0.001,
        WEIGHT_DECAY=0.0001,LR_WARMUP=False,DECAY_STEP_LIST=[],GRAD_NORM_CLIP=10)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(yaml.safe_dump(cfg,sort_keys=False))


if __name__=='__main__':main()
