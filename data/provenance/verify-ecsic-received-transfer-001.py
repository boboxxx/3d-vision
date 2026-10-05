"""Fresh actual-received file identities on each host, without neural execution."""
import argparse
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--host', choices=('artemis', 'local', 'sheng'), required=True)
    p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    assert not args.output.exists()
    index_path = ROOT / 'data/provenance/ecsic-received-transfer-index-001.json'
    index = read(index_path); manifest = ROOT / 'data/engineering/artemis-ecsic-digital-CPU-001.json'
    assert sha(manifest) == index['manifest_sha256']
    packets = read(manifest)['packets']; files = {}
    for x in packets:
        if x['reception']['state'] == 'received':
            for key in ('wire', 'payload'): files[x['received_paths'][key]] = x['received_paths'][key + '_sha256']
        else: assert x['received_paths'] is None and x['reception']['erasure_reason']
    assert files == index['files_sha256'] and len(files) == 22
    assert sum(x['reception']['state'] == 'received' for x in packets) == 11
    assert sum(x['reception']['state'] == 'erased' for x in packets) == 9
    assert all(sha(ROOT / path) == h for path, h in files.items())
    value = dict(state='passed_all22_actual_received_files', host=args.host, checked_unix=time.time(),
                 manifest_sha256=sha(manifest), index_sha256=sha(index_path), files_sha256=files,
                 received=11, erased=9, verifier_sha256=sha(Path(__file__)))
    with args.output.open('x') as stream: json.dump(value, stream, indent=2)
    print(json.dumps(dict(state=value['state'], host=args.host, files=22)))


if __name__ == '__main__': main()
