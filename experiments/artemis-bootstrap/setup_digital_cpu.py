"""Independent PHY-only CPU runtime; existing CUDA environments untouched."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def identities(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in ('.py', '.csv', '.toml')
            and not any(x in p.parts for x in ('build', '__pycache__'))}


def main():
    root = Path(__file__).resolve().parents[2]
    target = root / 'envs/digital-cpu-001'
    output = root / 'data/engineering/artemis-digital-runtime-001.json'
    source = root / 'dependencies/sionna-source-001'
    if target.exists() or output.exists() or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('allocated CPU and unique paths required')
    expected = json.loads((root / 'data/provenance/sionna-source-001.json').read_text())
    before = identities(source)
    if before != expected['files_sha256']:
        raise RuntimeError('pinned source mismatch')
    record = dict(state='starting',job_id=os.environ['SLURM_JOB_ID'],started_at_unix=time.time(),
                  environment=str(target),source_revision=expected['revision'],source_sha256=before,
                  scope='CPU_PHY_only_RT_deliberately_not_installed')
    with output.open('x') as f: json.dump(record,f,indent=2)
    try:
        subprocess.run([sys.executable,'-m','venv',target],check=True)
        python = str(target / 'bin/python')
        report1 = root / 'data/engineering/artemis-digital-torch-wheels-001.json'
        report2 = root / 'data/engineering/artemis-digital-base-wheels-001.json'
        subprocess.run([python,'-m','pip','install','--no-cache-dir','--report',report1,
                        'torch==2.9.1','--index-url','https://download.pytorch.org/whl/cpu'],check=True)
        subprocess.run([python,'-m','pip','install','--no-cache-dir','--report',report2,
                        'numpy==2.2.6','scipy==1.15.3','matplotlib==3.10.8','h5py==3.15.1',
                        'importlib-resources==6.5.2','setuptools==80.9.0','wheel==0.45.1'],check=True)
        subprocess.run([python,'-m','pip','install','--no-deps','--no-build-isolation',source],check=True)
        freeze = subprocess.check_output([python,'-m','pip','freeze'],text=True)
        freeze_path = root / 'data/engineering/artemis-digital-freeze-001.txt'
        with freeze_path.open('x') as f:f.write(freeze)
        check = subprocess.run([python,'-m','pip','check'],capture_output=True,text=True)
        record['pip_check'] = dict(returncode=check.returncode,stdout=check.stdout,stderr=check.stderr,
                                  deliberate_omission='sionna-rt; no ray tracing used')
        if check.returncode and check.stdout.strip() != 'sionna 2.1.0 requires sionna-rt, which is not installed.':
            raise RuntimeError('unexpected unresolved package dependencies')
        info = subprocess.check_output([python,'-c',
             'import json,torch,numpy,scipy,sionna,sionna.phy; print(json.dumps(dict(torch=torch.__version__,numpy=numpy.__version__,scipy=scipy.__version__,sionna=sionna.__version__,sionna_file=sionna.__file__)))'],text=True)
        record['CPU_imports'] = json.loads(info)
        assert record['CPU_imports']['torch']=='2.9.1+cpu' and record['CPU_imports']['sionna']=='2.1.0'
        installed = Path(record['CPU_imports']['sionna_file']).parent
        selected = {str(p.relative_to(source / 'src/sionna')): before[str(p.relative_to(source))]
                    for p in (source / 'src/sionna').rglob('*') if p.is_file() and p.suffix in ('.py','.csv')}
        if any(hashlib.sha256((installed / p).read_bytes()).hexdigest()!=v for p,v in selected.items()):
            raise RuntimeError('installed PHY source differs')
        if identities(source) != before:raise RuntimeError('source changed during install')
        record.update(state='passed_independent_CPU_PHY_imports_sources_exact_RT_omitted',
                      installed_source_sha256=selected,freeze_sha256=hashlib.sha256(freeze.encode()).hexdigest())
    except Exception as e:
        record.update(state='failed',error=f'{type(e).__name__}: {e}')
        raise
    finally:
        record['finished_at_unix']=time.time();output.write_text(json.dumps(record,indent=2)+'\n')
    print(record['state'])


if __name__=='__main__':main()
