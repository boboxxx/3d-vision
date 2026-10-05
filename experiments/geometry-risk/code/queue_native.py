"""One detached controller waits actual earlier jobs; it never restarts a native run."""
import os
from pathlib import Path
import subprocess
import sys
import time
from probe_native import ROOT,HERE,INPUT,read,save,sha256,terminal,sources,metadata,cpu_gate,validate_previous


def main():
    prefix='geometry-risk-native-engineering-001'
    path=ROOT/'data/runs'/f'{prefix}-queue.json';assert not path.exists(), 'preserve earlier controller'
    lock=read(INPUT);expected=sources();meta=metadata();cpu_sha=cpu_gate()
    record=dict(state='waiting_previous_actual_processes',prefix=prefix,pid=os.getpid(),started_unix=time.time(),
                input_sha256=sha256(INPUT),source_identities=expected,metadata_identities=meta,CPU_gate_sha256=cpu_sha,required_previous_PIDs=lock['wait_PIDs'])
    save(path,record)
    try:
        while True:
            assert sources()==expected and metadata()==meta and cpu_gate()==cpu_sha, 'frozen sources, protocol metadata or CPU gate changed while waiting'
            live=[pid for pid in lock['wait_PIDs'] if not terminal(pid)]
            record.update(previous_actual_live_PIDs=live,checked_unix=time.time());save(path,record)
            if not live:break
            time.sleep(15)
        validate_previous(lock)
        while True:
            assert sources()==expected and metadata()==meta
            used,total=map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(','))
            if (total-used)*2**20>=12*2**30:break
            record.update(state='waiting_physical_margin',physical_free_MiB=total-used,checked_unix=time.time());save(path,record);time.sleep(15)
        record.update(state='running_native_engineering');save(path,record)
        log=ROOT/'logs'/f'{prefix}-native.log'
        with log.open('x') as stream:
            subprocess.run([sys.executable,str(HERE/'probe_native.py'),'--prefix',prefix],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        assert sources()==expected and metadata()==meta
        report=read(ROOT/'data/engineering'/f'{prefix}.json')
        assert report['state']=='finished_native_engineering_audit_pending' and terminal(report['pid']) and report['metadata_identities']==meta
        record.update(state='native_engineering_terminal_independent_audit_pending',native_manifest_sha256=sha256(ROOT/'data/engineering'/f'{prefix}.json'),
                      native_log_sha256=sha256(log),ended_unix=time.time())
    except BaseException as exc:
        record.update(state='failed',exception=repr(exc),ended_unix=time.time());raise
    finally:save(path,record)


if __name__=='__main__':main()
