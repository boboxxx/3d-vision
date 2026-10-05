#!/usr/bin/env python3
"""Fixed codec tuning fold drawn exclusively from the official training set."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from geocomm.evidence import sha256


def split_ids(train, validation, holdout):
    if set(train)&set(validation) or len(set(train))!=len(train):
        raise ValueError('split overlap or duplicate')
    if not 0<holdout<len(train):
        raise ValueError('invalid holdout count')
    selected=set(sorted(train,key=lambda frame:hashlib.sha256(
        ('geocomm-codec-tuning-v1|'+frame).encode()).digest())[:holdout])
    return [x for x in train if x not in selected],[x for x in train if x in selected]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('preserve prior fold manifest')
    root=args.root.resolve()
    train_path,val_path=root/'ImageSets/train.txt',root/'ImageSets/val.txt'
    train,val=train_path.read_text().split(),val_path.read_text().split()
    assert len(train)==3712 and len(val)==3769
    training,holdout=split_ids(train,val,372)
    original=root/'kitti_infos_train.pkl'
    with original.open('rb') as stream:infos=pickle.load(stream)
    assert [x['point_cloud']['lidar_idx'] for x in infos]==train
    lookup=dict(zip(train,infos))
    names={'geocomm_tune_train':training,'geocomm_tune_holdout':holdout}
    targets=[root/'ImageSets'/(name+'.txt') for name in names]
    targets += [root/('kitti_infos_'+name+'.pkl') for name in names]
    if any(path.exists() for path in targets):parser.error('fold targets already exist')
    report=dict(state='finished',evidence_type='training_only_codec_tuning_fold',
        rule='smallest372 SHA256(geocomm-codec-tuning-v1|frame_id); retain original order',
        source_train_split_sha256=sha256(train_path),main_validation_split_sha256=sha256(val_path),
        source_train_infos_sha256=sha256(original),main_validation_used_for_tuning=False,
        limitation='released detector checkpoint was trained on original train; holdout isolates new codec training only',folds={})
    for name,ids in names.items():
        split=root/'ImageSets'/(name+'.txt');split.write_text('\n'.join(ids)+'\n')
        target=root/('kitti_infos_'+name+'.pkl');temporary=target.with_suffix('.pkl.part')
        with temporary.open('wb') as stream:pickle.dump([lookup[x] for x in ids],stream,protocol=4)
        temporary.replace(target)
        report['folds'][name]=dict(count=len(ids),ids=ids,split_sha256=sha256(split),infos_sha256=sha256(target))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({name:len(ids) for name,ids in names.items()}))


if __name__=='__main__':main()
