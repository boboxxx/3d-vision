"""Generate only author-defined version metadata and replay real registry imports."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUT=ROOT/'data/engineering/artemis-framework-import-006.json'
WORK=ROOT/'dependencies/artemis-framework-import-006'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def freeze():
    env=os.environ.copy();env.pop('PYTHONPATH',None)
    return subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True,env=env)
def main():
    assert not OUT.exists() and not WORK.exists();WORK.mkdir()
    lock=json.loads((HERE/'import-inputs-006.json').read_text())
    report=dict(state='running',stage='check_inputs',job_id=os.environ.get('SLURM_JOB_ID'),
        started_unix=time.time(),input_sha256=sha(HERE/'import-inputs-006.json'))
    def save():OUT.write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        got={p:sha(ROOT/p) for p in lock['frozen_inputs']};assert got==lock['frozen_inputs']
        report['sources_before']=got;report['base_freeze_before']=freeze()
        assert report['base_freeze_before']==lock['base_freeze']
        package=ROOT/'third_party/LIGA-Stereo';target=package/'liga/version.py'
        assert not (package/'.git').exists() and not target.exists()
        with target.open('x') as f:f.write('__version__ = "0.1.0+0000000"\n')
        report['generated_version_sha256']=sha(target);report['stage']='real_registry_import';save()
        env=os.environ.copy();toolkit=ROOT/'toolchains/cuda-12.8-001'
        env.update(PYTHONPATH=str(ROOT/'dependencies/artemis-framework-overlay-005/site-packages'),
            CUDA_HOME=str(toolkit),CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
        env['PATH']=str(toolkit/'bin')+os.pathsep+env['PATH']
        with (WORK/'real-import-probe.log').open('w') as f:
            completed=subprocess.run([sys.executable,str(HERE/'import-probe-005.py')],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        report['probe_returncode']=completed.returncode
        report['probe_log_sha256']=sha(WORK/'real-import-probe.log')
        report['probe_log']=(WORK/'real-import-probe.log').read_text()
        assert completed.returncode==0,'real registry import failed; retain complete log'
        report['state']='passed_real_framework_imports_pending_actual_terminal_GPU_task_unverified'
    except BaseException:
        report['state']='failed';report['traceback']=traceback.format_exc();raise
    finally:
        report['base_freeze_after']=freeze();report['base_runtime_unchanged']=report['base_freeze_after']==lock['base_freeze']
        report['sources_after']={p:sha(ROOT/p) for p in lock['frozen_inputs']}
        report['frozen_inputs_unchanged']=report['sources_after']==lock['frozen_inputs']
        report['ended_unix']=time.time();save()
        assert report['base_runtime_unchanged'] and report['frozen_inputs_unchanged']
if __name__=='__main__':main()
