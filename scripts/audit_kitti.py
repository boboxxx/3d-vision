#!/usr/bin/env python3
"""Verify stereo pairing, valid calibration and disjoint frozen split lists."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from PIL import Image


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(data_root, train_path, val_path):
    train = train_path.read_text(encoding="utf-8").split()
    val = val_path.read_text(encoding="utf-8").split()
    for name, ids in [("train", train), ("val", val)]:
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(name + " split is empty or duplicated")
        if any(re.fullmatch(r"\d{6}", frame) is None for frame in ids):
            raise ValueError("invalid KITTI id")
    if set(train) & set(val):
        raise ValueError("train/val overlap")
    training = data_root / "training"
    calibration_hash = hashlib.sha256()
    principal_offsets = []
    for frame in train + val:
        for folder, suffix in [("image_2", ".png"), ("image_3", ".png"),
                               ("calib", ".txt"), ("label_2", ".txt"),
                               ("velodyne", ".bin")]:
            path = training / folder / (frame + suffix)
            if not path.is_file() or (folder != "label_2" and path.stat().st_size == 0):
                raise FileNotFoundError(str(path))
        with Image.open(training / "image_2" / (frame + ".png")) as left:
            with Image.open(training / "image_3" / (frame + ".png")) as right:
                if left.size != right.size:
                    raise ValueError("stereo image size mismatch: " + frame)
                left.verify()
                right.verify()
        calib_path = training / "calib" / (frame + ".txt")
        matrices = {}
        for line in calib_path.read_text(encoding="utf-8").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                if key in ["P2", "P3"]:
                    matrices[key] = np.asarray([float(x) for x in value.split()]).reshape(3, 4)
        p2, p3 = matrices["P2"], matrices["P3"]
        baseline = abs(p3[0, 3]/p3[0, 0] - p2[0, 3]/p2[0, 0])
        if not np.isfinite(p2).all() or not np.isfinite(p3).all() or not .01 < baseline < 5:
            raise ValueError("invalid stereo calibration: " + frame)
        principal_offsets.append(float(p2[0, 2]-p3[0, 2]))
        calibration_hash.update(calib_path.read_bytes())
    return {"train_count": len(train), "val_count": len(val),
            "train_sha256": digest(train_path), "val_sha256": digest(val_path),
            "calibration_sha256": calibration_hash.hexdigest(),
            "max_principal_point_offset_pixels": max(abs(x) for x in principal_offsets),
            "status": "paired_files_and_calibration_verified",
            "image_content_hashes": "not_computed", "dataset_identity": "user_supplied_KITTI_object"}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--train", required=True, type=Path)
    p.add_argument("--val", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    report = audit(args.data_root, args.train, args.val)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
