#!/usr/bin/env python3
"""Download the standard released KITTI split at an immutable source revision."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = "233f849829b6ac19afb8af8837a0246890908755"
BASE = "https://raw.githubusercontent.com/open-mmlab/OpenPCDet/" + REVISION + "/data/kitti/ImageSets/"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    target = args.root / "ImageSets"
    target.mkdir(parents=True, exist_ok=True)
    ids, metadata, contents = {}, {}, {}
    for split, count in [("train", 3712), ("val", 3769)]:
        url = BASE + split + ".txt"
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
        rows = content.decode("utf-8").splitlines()
        if len(rows) != count or len(set(rows)) != count:
            raise ValueError("incorrect split count/duplicate: " + split)
        if any(len(row) != 6 or not row.isdigit() or int(row) >= 7481 for row in rows):
            raise ValueError("invalid frame id: " + split)
        ids[split] = set(rows)
        contents[split] = content
        metadata[split] = dict(url=url, count=count, sha256=hashlib.sha256(content).hexdigest())
    if ids["train"] & ids["val"] or ids["train"] | ids["val"] != {"%06d" % i for i in range(7481)}:
        raise ValueError("split partition invalid")
    for split, content in contents.items():
        path = target / (split + ".txt")
        if path.exists() and path.read_bytes() != content:
            raise ValueError("existing split differs; refusing replacement: " + str(path))
        path.write_bytes(content)
    (target / "split-provenance.json").write_text(json.dumps(dict(
        source_revision=REVISION, splits=metadata,
        note="Standard 3712/3769 KITTI split; original semantic-paper split still requires verification."
    ), indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
