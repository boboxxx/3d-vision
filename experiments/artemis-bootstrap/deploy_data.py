"""Complete public data deployment with sealed prior archive/split identities."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / 'data/engineering/artemis-data-deployment-001.json'
    target = root / 'data/kitti'
    if output.exists() or target.exists() or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('allocated CPU and unique deployment required')
    scripts = [root / 'scripts/download_kitti.py', root / 'scripts/prepare_kitti_splits.py']
    source = {p.name: sha(p) for p in scripts}
    reference_path = root / 'data/kitti-download-final.json'
    reference = json.loads(reference_path.read_text())
    if reference['state'] != 'finished' or len(reference['archives']) != 5:
        raise RuntimeError('sealed sheng archive reference required')
    record = dict(state='starting', job_id=os.environ['SLURM_JOB_ID'], started_at_unix=time.time(),
                  target=str(target), source_sha256=source, reference_sha256=sha(reference_path),
                  scope='complete_KITTI_deployment_only_no_training_AP')
    with output.open('x') as handle:
        json.dump(record, handle, indent=2)
    try:
        subprocess.run([sys.executable, scripts[0], '--root', target, '--workers', '2', '--segments', '4'], check=True)
        received = json.loads((target / 'download-manifest.json').read_text())
        if received['state'] != 'finished' or set(received['archives']) != set(reference['archives']):
            raise RuntimeError('complete downloaded archive set required')
        for name, expected in reference['archives'].items():
            actual = received['archives'][name]
            if any(actual[k] != expected[k] for k in ('bytes', 'sha256', 'training_files', 'training_member_identity_sha256')):
                raise RuntimeError('archive identity mismatch: ' + name)
            if actual['training_files'] != 7481 or sha(target / 'archives' / name) != expected['sha256']:
                raise RuntimeError('archive fresh SHA/count mismatch: ' + name)
        subprocess.run([sys.executable, scripts[1], '--root', target], check=True)
        expected_split = {'train': 'b6417a1d9b18c8fdb085128e633d28ff321b7674a6d1b3841b8f43d865b281cb',
                          'val': '657ac4bcc1e156e5b106a4ca18e1f88e012787ea1d2b5d0adeea97fee903fa86'}
        if any(sha(target / 'ImageSets' / (k + '.txt')) != v for k, v in expected_split.items()):
            raise RuntimeError('fixed split SHA mismatch')
        if any(sha(p) != source[p.name] for p in scripts) or sha(reference_path) != record['reference_sha256']:
            raise RuntimeError('data source or reference changed during deployment')
        record.update(state='passed_all5_archives_CRC_SHA_and_fixed3712_3769_splits',
                      archive_manifest_sha256=sha(target / 'download-manifest.json'),
                      splits_sha256=expected_split, archives=received['archives'])
    except Exception as error:
        record.update(state='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        record['finished_at_unix'] = time.time()
        output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
