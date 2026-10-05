#!/usr/bin/env python3
"""Audit the completed data and create upstream train/val infos, never test labels."""
import argparse
import json
from pathlib import Path
import pickle
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "third_party/LIGA-Stereo")]
from audit_kitti import audit
from geocomm.evidence import sha256, source_identity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    root = args.root.resolve()
    if args.output.exists():
        p.error("retain previous preparation manifest; choose unique output")
    download = root / "download-manifest.json"
    if json.loads(download.read_text())["state"] != "finished":
        p.error("full download/extraction must finish first")
    targets = {split: root / ("kitti_infos_" + split + ".pkl") for split in ["train", "val"]}
    if any(path.exists() for path in targets.values()):
        p.error("infos already exist; inspect their provenance before replacing")
    report = dict(state="running", started_at_unix=time.time(),
                  download_manifest_sha256=sha256(download))
    try:
        report["dataset_audit"] = audit(root, root / "ImageSets/train.txt", root / "ImageSets/val.txt")
        from easydict import EasyDict
        from liga.datasets.kitti.lidar_kitti_dataset import LiDARKittiDataset
        config_path = ROOT / "third_party/LIGA-Stereo/configs/stereo/dataset_configs/kitti_dataset_fused.yaml"
        config = EasyDict(yaml.safe_load(config_path.read_text()))
        dataset = LiDARKittiDataset(config, ["Car", "Pedestrian", "Cyclist"], root_path=root, training=False)
        report["upstream_sources"] = source_identity(ROOT / "third_party/LIGA-Stereo", ["liga", "configs", "tools", "setup.py"])
        report["infos"] = {}
        for split, target in targets.items():
            dataset.set_split(split)
            infos = dataset.get_infos(num_workers=args.workers, has_label=True, count_inside_pts=True)
            expected_ids = (root / "ImageSets" / (split + ".txt")).read_text().split()
            if [row["point_cloud"]["lidar_idx"] for row in infos] != expected_ids:
                raise RuntimeError("infos order/count differs from locked split")
            temporary = target.with_suffix(".pkl.part")
            with temporary.open("wb") as stream:
                pickle.dump(infos, stream, protocol=4)
            temporary.replace(target)
            report["infos"][split] = dict(count=len(infos), path=str(target), sha256=sha256(target))
        report["state"] = "finished"
    except BaseException as exc:
        report["state"], report["exception"] = "failed", repr(exc)
        raise
    finally:
        report["ended_at_unix"] = time.time()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
