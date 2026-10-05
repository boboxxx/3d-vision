"""Project-local, checksum-locked CUDA toolkit and unchanged LIGA operator build."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
INPUT = HERE.parent / 'inputs-001.json'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sources(lock):
    result = {name: sha(ROOT / name) for name in lock['source_files']}
    assert result == lock['source_files'], 'native source hash changed'
    return result


def run(argv):
    return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT)


def install_toolkit(lock):
    base = ROOT / lock['toolkit_directory']
    base.mkdir(parents=True, exist_ok=False)
    archives = base / 'archives'
    archives.mkdir()
    meta = archives / 'redistrib_12.8.1.json'
    urllib.request.urlretrieve(lock['metadata_url'], meta)
    assert sha(meta) == lock['metadata_sha256']
    upstream = json.loads(meta.read_text())
    records = {}
    for name, item in lock['components'].items():
        assert upstream[name]['linux-x86_64'] == {k: item[k] for k in ('relative_path', 'sha256', 'md5', 'size')}
        archive = archives / Path(item['relative_path']).name
        urllib.request.urlretrieve('https://developer.download.nvidia.com/compute/cuda/redist/' + item['relative_path'], archive)
        assert archive.stat().st_size == int(item['size']) and sha(archive) == item['sha256']
        component = base / 'components' / name
        component.mkdir(parents=True)
        with tarfile.open(archive, 'r:xz') as tf:
            tf.extractall(component, filter='data')
        extracted = list(component.iterdir())
        assert len(extracted) == 1 and extracted[0].is_dir()
        # Keep each complete component, including license, and assemble the toolkit tree.
        for tree in ('bin', 'include', 'lib', 'lib64', 'nvvm', 'share'):
            src = extracted[0] / tree
            if not src.is_dir():
                continue
            target = base / tree
            for p in src.rglob('*'):
                out = target / p.relative_to(src)
                if p.is_dir():
                    out.mkdir(parents=True, exist_ok=True)
                elif p.is_symlink():
                    out.parent.mkdir(parents=True, exist_ok=True)
                    if out.is_symlink():
                        assert os.readlink(out) == os.readlink(p)
                    else:
                        assert not out.exists()
                        out.symlink_to(os.readlink(p))
                else:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    if out.exists():
                        assert sha(out) == sha(p), 'conflicting toolkit component file'
                    else:
                        shutil.copy2(p, out)
        records[name] = {'archive_sha256': sha(archive), 'bytes': archive.stat().st_size}
    if not (base / 'lib64').exists():
        assert (base / 'lib').is_dir()
        (base / 'lib64').symlink_to('lib')
    assert (base / 'bin/nvcc').is_file() and (base / 'include/cuda.h').is_file()
    assert (base / 'lib64/libcudart.so').exists()
    return base, records


def main():
    output = ROOT / 'data/engineering/artemis-detector-operators-build-001.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    assert not output.exists(), 'preserve earlier attempt'
    lock = json.loads(INPUT.read_text())
    report = {'state': 'running', 'job_id': os.environ.get('SLURM_JOB_ID'), 'started_unix': time.time(),
              'input_sha256': sha(INPUT), 'protocol_sha256': sha(HERE.parent / 'protocol-001.md'),
              'execution_sources': {str(p.relative_to(ROOT)): sha(p) for p in HERE.glob('*.py')}}
    try:
        report['sources_before'] = sources(lock)
        report['packages_before'] = run([sys.executable, '-m', 'pip', 'freeze'])
        toolkit, report['components'] = install_toolkit(lock)
        report['nvcc_version'] = run([str(toolkit / 'bin/nvcc'), '--version'])
        report['gcc_version'] = run(['g++', '--version'])
        build = ROOT / lock['build_directory']
        build.mkdir(parents=True, exist_ok=False)
        os.environ['CUDA_HOME'] = str(toolkit)
        os.environ['TORCH_CUDA_ARCH_LIST'] = lock['architecture']
        os.environ['MAX_JOBS'] = '4'
        os.environ['PATH'] = str(toolkit / 'bin') + os.pathsep + os.environ['PATH']
        report['build_stdout'] = run([sys.executable, str(HERE / 'setup_build.py'), 'build_ext',
                                     '--build-lib', str(build / 'lib'), '--build-temp', str(build / 'temp'), '--parallel', '2'])
        report['binaries'] = {str(p.relative_to(ROOT)): {'sha256': sha(p), 'bytes': p.stat().st_size}
                              for p in sorted((build / 'lib').glob('*.so'))}
        assert len(report['binaries']) == 3, 'all three extensions required'
        report['sources_after'] = sources(lock)
        report['packages_after'] = run([sys.executable, '-m', 'pip', 'freeze'])
        assert report['packages_before'] == report['packages_after'], 'runtime packages changed'
        report['state'] = 'passed_build_only_GPU_unverified'
    except BaseException as e:
        report['state'] = 'failed'
        report['traceback'] = traceback.format_exc()
        if isinstance(e, subprocess.CalledProcessError):
            report['failed_command_output'] = e.output
        raise
    finally:
        report['ended_unix'] = time.time()
        output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: report[k] for k in ('state', 'job_id')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
