"""Read-only package resolution; no installation or model/GPU execution."""
import hashlib
import html.parser
import json
from pathlib import Path
import re
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'data/engineering/artemis-framework-resolution-001'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class Links(html.parser.HTMLParser):
    def __init__(self):
        super().__init__(); self.values = []
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.values.extend(v for k, v in attrs if k == 'href')

def main():
    OUT.mkdir(exist_ok=False)
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (OUT / 'base-freeze.txt').write_text(freeze)
    index_url = 'https://download.pytorch.org/whl/cu128/torchvision/'
    index = urllib.request.urlopen(index_url, timeout=90).read()
    (OUT / 'torchvision-index.html').write_bytes(index)
    parser = Links(); parser.feed(index.decode())
    links = [v for v in parser.values if re.search(r'torchvision-0\.22\.1(?:%2B|\+)cu128-cp312-cp312-manylinux_2_28_x86_64\.whl', v)]
    assert len(links) == 1, links
    selected = OUT / 'requirements.txt'
    selected.write_text((HERE / 'requirements-001.txt').read_text() + '\ntorchvision @ ' + links[0] + '\n')
    report = OUT / 'pip-report.json'
    cmd = [sys.executable, '-m', 'pip', 'install', '--dry-run', '--only-binary=:all:',
           '--no-build-isolation', '--report', str(report), '-c', str(OUT / 'base-freeze.txt'),
           '-r', str(selected)]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (OUT / 'resolution.log').write_text(result.stdout)
    if result.returncode:
        (OUT / 'status.json').write_text(json.dumps({'state': 'failed_resolution', 'returncode': result.returncode}))
        raise RuntimeError('resolution failed; complete output retained')
    resolved = json.loads(report.read_text())
    existing = {re.sub('[-_.]+', '-', v.split('==')[0]).lower() for v in freeze.splitlines() if '==' in v}
    archives = []
    for item in resolved['install']:
        name = re.sub('[-_.]+', '-', item['metadata']['name']).lower()
        assert name not in existing and name != 'torch' and not name.startswith('nvidia-'), name
        info = item['download_info']; url = info['url']
        assert url.startswith(('https://files.pythonhosted.org/', 'https://download.pytorch.org/')), url
        digest = info['archive_info']['hashes']['sha256']; assert len(digest) == 64
        archives.append(dict(name=name, version=item['metadata']['version'], url=url, sha256=digest))
    lock = dict(state='resolved_only_not_installed', base_freeze=freeze, archives=archives,
                protocol_sha256=sha(HERE / 'protocol-001.md'), requirements_sha256=sha(HERE / 'requirements-001.txt'),
                resolver_sha256=sha(__file__), torchvision_index_sha256=hashlib.sha256(index).hexdigest(),
                mmcv=dict(name='mmcv-full', version='1.7.2',
                    url='https://files.pythonhosted.org/packages/c9/cc/21ba65de872f57418d8b233c718cddb6f55f86ef1aa399141344887c052c/mmcv-full-1.7.2.tar.gz',
                    sha256='aa276a2c68fa84db9bbcdcf6ae941e30f0cb4a7f175bd283516669e00bab3759'))
    (OUT / 'resolved-lock.json').write_text(json.dumps(lock, indent=2) + '\n')
    print(json.dumps({'state': lock['state'], 'additional_archives': len(archives)}))

if __name__ == '__main__':
    main()
