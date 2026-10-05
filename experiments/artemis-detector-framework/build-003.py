"""Checksum-locked project overlay and unchanged MMCV source build, CPU Slurm."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
INPUT = HERE / 'build-inputs-003.json'
WORK = ROOT / 'dependencies/artemis-framework-overlay-003'
OUT = ROOT / 'data/engineering/artemis-framework-build-003.json'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def freeze():
    env = os.environ.copy(); env.pop('PYTHONPATH', None)
    return subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True, env=env)

def run(cmd, name, env=None, cwd=None):
    with (WORK / (name + '.log')).open('w') as out:
        result = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, env=env, cwd=cwd)
    assert result.returncode == 0, f'{name}: exit {result.returncode}; see retained complete log'

def sources(lock):
    got = {p: sha(ROOT / p) for p in lock['frozen_inputs']}
    assert got == lock['frozen_inputs'], 'frozen input changed'
    return got

def download(item):
    name = urllib.parse.unquote(Path(urllib.parse.urlsplit(item['url']).path).name)
    assert name and '/' not in name and name not in ('.', '..')
    path = ROOT / 'dependencies/artemis-framework-archives-003' / name
    assert path.is_file() and sha(path) == item['sha256'], name + ': staged archive hash'
    return path


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'CPU Slurm allocation required'
    assert not OUT.exists() and not WORK.exists(), 'preserve prior attempt'
    lock = json.loads(INPUT.read_text()); WORK.mkdir(); (WORK / 'archives').mkdir()
    report = dict(state='running', stage='input_checks', job_id=os.environ['SLURM_JOB_ID'],
                  started_unix=time.time(), input_sha256=sha(INPUT))
    def save(): OUT.write_text(json.dumps(report, indent=2) + '\n')
    save()
    try:
        report['sources_before'] = sources(lock)
        report['base_packages_before'] = freeze()
        assert report['base_packages_before'] == lock['resolution']['base_freeze']
        report['stage'] = 'verify_offline_archives'; save()
        wheels = [download(v) for v in lock['resolution']['archives']]
        source = download(lock['resolution']['mmcv'])
        report['archive_sha256'] = {str(p.relative_to(ROOT)): sha(p) for p in wheels + [source]}
        target = WORK / 'site-packages'
        run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-index',
             '--target', str(target), *map(str, wheels)], 'install-overlay')
        env = os.environ.copy(); toolkit = ROOT / lock['toolkit_directory']
        env.update(PYTHONPATH=str(target), CUDA_HOME=str(toolkit), FORCE_CUDA='1',
                   TORCH_CUDA_ARCH_LIST='12.0', MAX_JOBS='4', MMCV_WITH_OPS='1',
                   PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', CUDA_VISIBLE_DEVICES='')
        env['PATH'] = str(toolkit / 'bin') + os.pathsep + env['PATH']
        report['stage'] = 'mmcv_unchanged_source_compile'; save()
        with tarfile.open(source) as archive:
            archive.extractall(WORK / 'source', filter='data')
        source_dir = WORK / 'source/mmcv-full-1.7.2'
        run([sys.executable, 'setup.py', 'bdist_wheel', '--dist-dir', str(WORK / 'built-wheels')],
            'mmcv-build', env=env, cwd=source_dir)
        built = list((WORK / 'built-wheels').glob('*.whl')); assert len(built) == 1
        report['built_wheel'] = dict(path=str(built[0].relative_to(ROOT)), sha256=sha(built[0]))
        run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-index',
             '--target', str(target), str(built[0])], 'install-mmcv')
        report['stage'] = 'real_registry_import'; save()
        run([sys.executable, str(HERE / 'import-probe-003.py')], 'real-import-probe', env=env, cwd=ROOT)
        report['import_probe_output'] = (WORK / 'real-import-probe.log').read_text()
        report['state'] = 'passed_framework_build_and_real_imports_pending_actual_terminal'
    except BaseException:
        report['state'] = 'failed'; report['traceback'] = traceback.format_exc()
        raise
    finally:
        report['base_packages_after'] = freeze()
        report['base_runtime_unchanged'] = report['base_packages_after'] == lock['resolution']['base_freeze']
        try:
            report['sources_after'] = sources(lock)
            report['frozen_inputs_unchanged'] = True
        except BaseException:
            report['frozen_inputs_unchanged'] = False
            report['source_check_failure'] = traceback.format_exc()
        if not report['base_runtime_unchanged'] or not report['frozen_inputs_unchanged']:
            report['state'] = 'failed_input_preservation'
        report['logs'] = {str(p.relative_to(ROOT)): sha(p) for p in WORK.glob('*.log')}
        report['ended_unix'] = time.time(); save()
        print(json.dumps({k: report[k] for k in ('state', 'stage', 'job_id')}), flush=True)
    assert report['state'] == 'passed_framework_build_and_real_imports_pending_actual_terminal'

if __name__ == '__main__': main()
