#!/usr/bin/env python3
"""Save exact deployed source identities alongside upstream repository revisions."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from geocomm.evidence import revision, source_identity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    trees = {"project": (ROOT, ["src", "scripts", "configs", "pyproject.toml"]),
             "liga": (ROOT / "third_party/LIGA-Stereo", ["liga", "configs", "tools", "setup.py"]),
             "mmdet": (ROOT / "third_party/mmdetection_kitti", ["mmdet"]),
             "stereo_rcnn": (ROOT / "third_party/Stereo-RCNN", ["lib", "demo.py", "test_net.py"])}
    output = {}
    for name, (root, paths) in trees.items():
        output[name] = dict(git_revision=revision(root), actual_sources=source_identity(root, paths))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({name: row["actual_sources"]["sha256"] for name, row in output.items()}, indent=2))


if __name__ == "__main__":
    main()
