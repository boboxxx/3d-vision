"""Fresh native hashes/runtime and actual Slurm GPU-probe terminal evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/mnt/nfs2/engdes/wc296/paper6')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output=ROOT/'data/engineering/artemis-detector-operators-GPU-terminal-001.json';assert not output.exists()
    report_path=ROOT/'data/engineering/artemis-detector-operators-GPU-001.json'
    report=json.loads(report_path.read_text());assert report['state']=='passed_native_operator_GPU_only'
    assert report['job_id']=='11423980' and report['capability']==[12,0] and 'RTX PRO 6000' in report['GPU']
    raw=subprocess.check_output(['sacct','-j','11423980','--format=JobID,State,ExitCode,AllocTRES,NodeList','-n','-P'],text=True)
    jobs=[]
    for line in raw.splitlines():
        fields=line.split('|');assert len(fields)>=5
        jobs.append(dict(zip(('JobID','State','ExitCode','AllocTRES','NodeList'),fields[:5])))
    assert {row['JobID'] for row in jobs}=={'11423980','11423980.batch','11423980.extern','11423980.0'}
    assert all(row['State']=='COMPLETED' and row['ExitCode']=='0:0' for row in jobs)
    assert all(row['NodeList']=='artemis-rtx-03' for row in jobs)
    freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    assert freeze==report['packages_before']==report['packages_after']
    build_path=ROOT/'data/engineering/artemis-detector-operators-build-001.json'
    assert sha(build_path)==report['build_manifest_sha256']
    lock_path=ROOT/'experiments/artemis-detector-operators/inputs-001.json'
    assert sha(lock_path)==report['input_sha256']
    lock=json.loads(lock_path.read_text())
    assert report['sources_before']==report['sources_after']==lock['source_files']
    fresh={str(path.relative_to(ROOT)):sha(path) for path in (build_path,lock_path)}
    for path,digest in {**report['execution_sources'],**report['sources_before']}.items():
        assert sha(ROOT/path)==digest;fresh[path]=digest
    for path,item in report['binaries'].items():
        assert sha(ROOT/path)==item['sha256'] and (ROOT/path).stat().st_size==item['bytes'];fresh[path]=item['sha256']
    artifacts={str(report_path.relative_to(ROOT)):sha(report_path)}
    folder=report_path.with_suffix('')
    assert {path.name for path in folder.glob('*.npz')}==set(report['artifacts'])
    for name,item in report['artifacts'].items():
        path=folder/name;assert sha(path)==item['sha256'] and path.stat().st_size==item['bytes']
        artifacts[str(path.relative_to(ROOT))]=sha(path)
    for extension in ('out','err'):
        path=ROOT/'logs'/('artemis-detector-operators-GPU-001-11423980.'+extension)
        artifacts[str(path.relative_to(ROOT))]=sha(path)
    assert len(artifacts)==6
    result=dict(state='closed_actual_Slurm_terminal_native_operator_GPU_probe',job_id='11423980',actual_terminal=True,
        checked_unix=time.time(),jobs=jobs,GPU=report['GPU'],capability=report['capability'],GPU_allocation_released=True,
        fresh_file_sha256=fresh,artifacts_sha256=artifacts,closure_script_sha256=sha(__file__),
        limitation='Three isolated native operator extensions/fixtures only; full detector, MMCV/spconv, rotated/boundary variants and AP unverified')
    with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(state=result['state'],job_id='11423980',artifacts=6)))


if __name__=='__main__':main()
