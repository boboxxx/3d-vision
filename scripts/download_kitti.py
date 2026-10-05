#!/usr/bin/env python3
"""Retrieve publicly readable KITTI object archives; extract full training set.

Preserves archive identities and validates ZIP CRCs while extracting. No test
labels are available. Use the official dataset terms for research use.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile
import zlib

BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
ARCHIVES = ["data_object_label_2.zip", "data_object_calib.zip",
            "data_object_image_2.zip", "data_object_image_3.zip",
            "data_object_velodyne.zip"]


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def remaining_ranges(offset, size, segment_bytes):
    """Keep an already downloaded prefix; subsequent ranges are disjoint."""
    return [(begin,min(begin+segment_bytes,size)-1) for begin in range(offset,size,segment_bytes)]


def parallel_remaining(url, target, part, size, etag, workers, segment_bytes):
    offset = part.stat().st_size if part.exists() else 0
    if offset > size:
        raise RuntimeError('partial file exceeds archive size')
    if offset == size:
        return
    directory = target.with_suffix('.zip.segments')
    directory.mkdir(exist_ok=True)
    identity = dict(url=url,size=size,etag=etag)
    identity_path = directory/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
        raise RuntimeError('segment archive identity changed; refusing to mix bytes')
    identity_path.write_text(json.dumps(identity,sort_keys=True))

    def fetch(bounds):
        begin,end = bounds
        destination = directory/f'{begin}-{end}.bin'
        checksum = destination.with_suffix('.json')
        length = end-begin+1
        if destination.exists() and checksum.exists():
            saved = json.loads(checksum.read_text())
            if destination.stat().st_size==length and sha256(destination)==saved['sha256']:
                return destination,bounds
        temporary = destination.with_suffix('.part')
        for attempt in range(10):
            try:
                request = urllib.request.Request(url,headers={'Range':f'bytes={begin}-{end}'})
                with urllib.request.urlopen(request,timeout=120) as response:
                    if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {begin}-{end}/{size}':
                        raise RuntimeError('server did not honor exact byte range')
                    if response.headers.get('ETag','').strip('"')!=etag:
                        raise RuntimeError('archive identity changed during segment download')
                    with temporary.open('wb') as stream:
                        shutil.copyfileobj(response,stream,length=1024*1024)
                if temporary.stat().st_size!=length:
                    raise OSError('incomplete segment')
                temporary.replace(destination)
                checksum.write_text(json.dumps(dict(sha256=sha256(destination))))
                return destination,bounds
            except (OSError,TimeoutError) as error:
                print(json.dumps(dict(archive=target.name,range=[begin,end],retry=attempt+1,error=repr(error))),flush=True)
                time.sleep(min(30,2**attempt))
        raise RuntimeError('segment download failed')

    ranges = remaining_ranges(offset,size,segment_bytes)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Ordered assembly preserves a contiguous resumable prefix even on interruption.
        for destination,(begin,end) in pool.map(fetch,ranges):
            current = part.stat().st_size if part.exists() else 0
            if current!=begin:
                raise RuntimeError('partial archive prefix changed during assembly')
            with destination.open('rb') as source,part.open('ab') as stream:
                shutil.copyfileobj(source,stream,length=1024*1024)
            if part.stat().st_size!=end+1:
                raise RuntimeError('assembled prefix size mismatch')
            destination.unlink()
            destination.with_suffix('.json').unlink()
    if part.stat().st_size!=size:
        raise RuntimeError('incomplete assembled archive')


def download(url, target, segments=1, segment_bytes=64*1024*1024):
    if segments<1 or segment_bytes<1:
        raise ValueError('positive segment count and size required')
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        size = int(response.headers["Content-Length"])
        etag = response.headers.get("ETag", "").strip('"')
    part = target.with_suffix(".zip.part")
    if not target.exists() or target.stat().st_size != size:
        if segments>1:
            parallel_remaining(url,target,part,size,etag,segments,segment_bytes)
        for attempt in range(10 if segments==1 else 0):
            offset = part.stat().st_size if part.exists() else 0
            if offset > size:
                raise RuntimeError("partial file exceeds expected archive size")
            if offset == size:
                break
            headers = {"Range": "bytes=%d-" % offset} if offset else {}
            try:
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=120) as response:
                    if offset and response.status != 206:
                        raise RuntimeError("server did not honor resume range")
                    if response.headers.get("ETag", "").strip('"') != etag:
                        raise RuntimeError("archive identity changed during download")
                    with part.open("ab" if offset else "wb") as output:
                        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
                if part.stat().st_size == size:
                    break
            except (OSError, TimeoutError) as exc:
                print(json.dumps({"archive": target.name, "retry": attempt + 1,
                                  "error": repr(exc)}), flush=True)
                time.sleep(min(30, 2 ** attempt))
        if not part.exists() or part.stat().st_size != size:
            raise RuntimeError("incomplete download: " + target.name)
        part.replace(target)
    # Single-part ETags are content MD5; multipart ETags are not content hashes.
    if "-" not in etag and len(etag) == 32:
        md5 = hashlib.md5()
        with target.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                md5.update(chunk)
        if md5.hexdigest() != etag:
            raise RuntimeError("S3 single-part checksum mismatch")
    return {"url": url, "bytes": size, "etag": etag,
            "sha256": sha256(target), "etag_is_content_md5": "-" not in etag}


def extract_training(archive, root):
    count = 0
    manifest = hashlib.sha256()
    with zipfile.ZipFile(archive) as package:
        for item in package.infolist():
            name = Path(item.filename)
            if not item.filename.startswith("training/") or item.is_dir():
                continue
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("unsafe archive member")
            output = root / name
            output.parent.mkdir(parents=True, exist_ok=True)
            current_crc = None
            if output.exists() and output.stat().st_size == item.file_size:
                crc = 0
                with output.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        crc = zlib.crc32(chunk, crc)
                current_crc = crc & 0xffffffff
            if current_crc != item.CRC:
                temporary = output.with_suffix(output.suffix + ".part")
                with package.open(item) as source, temporary.open("wb") as destination:
                    # Reading through EOF verifies the member's stored CRC.
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
                if temporary.stat().st_size != item.file_size:
                    raise RuntimeError("member size mismatch")
                temporary.replace(output)
            count += 1
            manifest.update((item.filename + ":%d:%08x\n" % (item.file_size, item.CRC)).encode("utf-8"))
    return {"training_files": count, "training_member_identity_sha256": manifest.hexdigest(),
            "validation": "ZIP_CRC_and_file_size"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument('--segments',type=int,default=1,
                        help='parallel ranged connections per archive; preserves existing prefix')
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    lock = (args.root/'download.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('this data root already has a live downloader')
    if shutil.disk_usage(args.root).free < 100 * 1024 ** 3:
        raise RuntimeError("need at least 100 GiB free for archives and full training data")
    archives = args.root / "archives"
    archives.mkdir(exist_ok=True)
    report = {"dataset": "KITTI_object", "source": BASE,
              'archive_workers':args.workers,'range_connections_per_archive':args.segments,
              "started_unix": time.time(), "state": "running", "archives": {}}
    report_path = args.root / "download-manifest.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def work(name):
        archive = archives / name
        identity = download(BASE + name, archive,segments=args.segments)
        print(json.dumps({"downloaded": name, **identity}), flush=True)
        identity.update(extract_training(archive, args.root))
        print(json.dumps({"extracted": name, **identity}), flush=True)
        return name, identity

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for name, identity in pool.map(work, ARCHIVES):
                report["archives"][name] = identity
                report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        counts = {folder: len(list((args.root / "training" / folder).glob("*")))
                  for folder in ["image_2", "image_3", "calib", "label_2", "velodyne"]}
        if any(count != 7481 for count in counts.values()):
            raise RuntimeError("full training set count mismatch: " + str(counts))
        report["training_counts"] = counts
        report["state"] = "finished"
    except BaseException as exc:
        report["state"], report["error"] = "failed", repr(exc)
        raise
    finally:
        report["last_updated_unix"] = time.time()
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
