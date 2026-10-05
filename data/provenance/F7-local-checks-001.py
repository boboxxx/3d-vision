"""Record meaningful CPU regressions and exact sources; no remote/native-GPU claim."""
import hashlib,json,os,re,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from geocomm.evidence import source_identity,sha256
HERE=ROOT/'experiments/geometry-link/F7/code';sys.path.insert(0,str(HERE))
from conditions import CONFIG_SHA,PROTOCOL_SHA,schedule_manifest
output=ROOT/'data/engineering/F7-local-regression-001.json';log=output.with_suffix('.log')
if output.exists() or log.exists():raise RuntimeError('retain prior local evidence')
assert sha256(ROOT/'configs/tuning/stereo_task_seed17_epoch1.yaml')==CONFIG_SHA
assert sha256(ROOT/'experiments/geometry-link/F7-matched-channel-adaptation.md')==PROTOCOL_SHA
specs={'project':(ROOT,['src','scripts','configs','pyproject.toml']),
 'liga':(ROOT/'third_party/LIGA-Stereo',['liga','configs','tools','setup.py']),
 'mmdet':(ROOT/'third_party/mmdetection_kitti',['mmdet']),
 'stereo_rcnn':(ROOT/'third_party/Stereo-RCNN',['lib','demo.py','test_net.py']),
 'experiment':(ROOT,['experiments/geometry-link/F7/code'])}
sources={k:source_identity(*v) for k,v in specs.items()};env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
commands=[];tests=0
with log.open('x') as stream:
 for relative in ('experiments/geometry-link/F7/code/test_conditions.py','experiments/geometry-link/F7/code/test_native_scope.py','tests/test_stereo_task_adaptation.py'):
  args=[sys.executable,str(ROOT/relative),'-v'];result=subprocess.run(args,cwd=ROOT,env=env,capture_output=True,text=True)
  stream.write(json.dumps(dict(command=relative,returncode=result.returncode))+'\n'+result.stdout+result.stderr);stream.flush()
  if result.returncode:raise RuntimeError('CPU regression failed: '+relative)
  match=re.search(r'Ran (\d+) tests?',result.stderr);assert match;count=int(match.group(1));tests+=count
  commands.append(dict(command=relative,returncode=0,tests=count))
 for file in HERE.glob('*.py'):compile(file.read_text(),str(file),'exec')
 result=subprocess.run([sys.executable,str(HERE/'cycle.py'),'--help'],cwd=ROOT,env=env,capture_output=True,text=True)
 stream.write(result.stdout+result.stderr);assert result.returncode==0
assert tests==13
for k,v in specs.items():assert source_identity(*v)==sources[k]
original=json.loads((ROOT/'data/provenance/cao2025-staged-trainer-source-bfee9d8.json').read_text())
assert len(original)==21 and all(sha256(ROOT/'reproduction/cao2025'/k)==v for k,v in original.items())
assert sources['project']['sha256']=='42b4b52a174f9c2a8c365ea45b3477e87d4c8165d3254d6b04b97bd2dad74b21'
record=dict(state='passed',checked_at_unix=time.time(),scope='local CPU fixture regressions only',tests=tests,
 commands=commands,syntax_checks='all F7 code files compiled',cycle_CLI_help='passed',protocol_sha256=PROTOCOL_SHA,
 config_sha256=CONFIG_SHA,schedule_sha256=schedule_manifest()['sha256'],sources=sources,original21_unchanged=True,
 log_sha256=sha256(log),limitations='No deployment, native GPU update, paired native engineering, trained model, or AP result. Native probe still requires Tailscale authentication.')
output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='sources'}))
