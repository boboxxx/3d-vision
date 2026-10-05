"""Login-node official archive staging only; no install/compile/import/GPU."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'data/engineering/artemis-framework-archives-004.json'
CACHE=ROOT/'dependencies/artemis-framework-archives-004'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for blob in iter(lambda:f.read(8*1024*1024),b''):h.update(blob)
    return h.hexdigest()

def main():
    assert not os.environ.get('SLURM_JOB_ID'), 'login-node staging, not compilation'
    assert not OUT.exists() and not CACHE.exists(), 'preserve earlier attempt'
    lock=json.loads((HERE/'build-inputs-004.json').read_text())
    CACHE.mkdir(); report=dict(state='running',started_unix=time.time(),archives=[])
    def save():OUT.write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        for item in lock['resolution']['archives']+[lock['resolution']['mmcv']]:
            url=item['url']; name=urllib.parse.unquote(Path(urllib.parse.urlsplit(url).path).name)
            assert name and '/' not in name and name not in ('.','..')
            path=CACHE/name; old=ROOT/'dependencies/artemis-framework-archives-003'/name
            report['current_archive']=name;save()
            if old.is_file() and sha(old)==item['sha256']:
                shutil.copyfile(old,path); origin='copied_verified_prior_archive'
            else:
                with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"pip/24.3.1"}),timeout=180) as inp,path.open('xb') as out:
                    while True:
                        blob=inp.read(8*1024*1024)
                        if not blob:break
                        out.write(blob)
                origin='official_download'
            assert sha(path)==item['sha256'],name
            report['archives'].append(dict(path=str(path.relative_to(ROOT)),url=url,
                sha256=item['sha256'],bytes=path.stat().st_size,origin=origin))
            save();print(json.dumps(dict(checked=len(report['archives']),archive=name)),flush=True)
        assert len(report['archives'])==48
        report['state']='staged_all48_locked_official_archives_no_installation'
    except BaseException:
        report['state']='failed';report['traceback']=traceback.format_exc();raise
    finally:
        report['ended_unix']=time.time();save()

if __name__=='__main__':main()
