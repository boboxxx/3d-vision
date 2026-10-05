"""Unique detached launch of the prelocked twelve-update F8 engineering pair."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];prefix='stereo-encoder-native-sanity-001'
p=ROOT/'data/runs'/f'{prefix}-launch.json';log=ROOT/'logs'/f'{prefix}-cycle.log'
assert not p.exists() and not log.exists() and not (ROOT/'data/runs'/f'{prefix}-cycle.json').exists()
sys.path.insert(0,str(ROOT/'experiments/geometry-link/F8/code'))
from eligibility import verify_eligibility
eligibility=verify_eligibility()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
gate=ROOT/'data/engineering/F8-sheng-CPU-003.json';cpu=json.loads(gate.read_text())
assert cpu['state']=='passed' and cpu['tests']==7
assert cpu['source_sha256']=={str(x.relative_to(ROOT)):sha(x) for x in (ROOT/'experiments/geometry-link/F8/code').glob('*.py')}
command=[sys.executable,str(ROOT/'experiments/geometry-link/F8/code/cycle.py'),'--prefix',prefix,'--engineering']
with log.open('x') as stream:
    child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
record=dict(pid=child.pid,started_unix=time.time(),prefix=prefix,command=command,engineering=True,eligibility=eligibility,CPU_gate_sha256=sha(gate),source_sha256=cpu['source_sha256'],launcher_sha256=sha(Path(__file__)))
with p.open('x') as out:json.dump(record,out,indent=2)
print(json.dumps({'pid':child.pid,'launch':str(p)}))
