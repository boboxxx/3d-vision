#!/usr/bin/env python3
"""Keep the entire upstream detector configuration, adding only link options."""
import argparse
from pathlib import Path
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument('--representation',choices=['geometry','rgb'],default='geometry')
    p.add_argument('--boundary',choices=['processed_cost','raw_cost'],default='processed_cost')
    p.add_argument('--student',action='store_true',help='optional distilled sender on raw-cost geometry')
    p.add_argument('--task-sensitivity',action='store_true')
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allocation", choices=["geometry", "geometry_task", "task", "uniform", "learned"], default="geometry")
    p.add_argument("--channel", choices=["awgn", "rayleigh", "identity"], default="awgn")
    p.add_argument("--width", type=int, default=4)
    p.add_argument("--snr", type=float, default=10.)
    args = p.parse_args()
    if args.task_sensitivity and args.representation != 'geometry':
        p.error('task sensitivity requires geometry representation')
    if args.allocation in ('geometry_task', 'task') and not args.task_sensitivity:
        p.error('task allocation requires --task-sensitivity')
    if args.student and (args.representation!='geometry' or args.boundary!='raw_cost'):
        p.error('--student requires geometry representation and raw_cost boundary')
    cfg = yaml.safe_load(args.base.read_text(encoding="utf-8"))
    options = {
        "enabled": True, "complex_width": args.width,
        "snr_db": args.snr, "channel": args.channel, "pilots": 8,
        "allocation": args.allocation, "posterior_weight": 0.1,
        "train_snr_min": -5., "train_snr_max": 20.,
        "geometry_stride": [8, 4, 4], "appearance_stride": 4,
    }
    if args.representation=='rgb':
        cfg['MODEL']['BACKBONE_3D'].pop('SEMANTIC_LINK',None)
        cfg['MODEL']['RGB_LINK'] = {key:value for key,value in options.items()
            if key not in {'allocation','posterior_weight','geometry_stride','appearance_stride'}}
    else:
        cfg['MODEL'].pop('RGB_LINK',None)
        options['boundary'] = args.boundary
        options['posterior_hidden'] = 16 if args.boundary=='raw_cost' else 0
        if args.task_sensitivity:
            options.update(task_sensitivity=True, sensitivity_weight=0.1)
        cfg["MODEL"]["BACKBONE_3D"]["SEMANTIC_LINK"] = options
    cfg['MODEL']['BACKBONE_3D'].pop('STUDENT_ENCODER',None)
    if args.student:
        cfg['MODEL']['BACKBONE_3D']['STUDENT_ENCODER'] = dict(enabled=True,distill_weight=0.1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
