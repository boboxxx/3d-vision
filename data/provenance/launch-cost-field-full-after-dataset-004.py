"""Current-task dependency: existing dataset audit -> full training, once only."""
import hashlib,json,os,re,shlex,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
REMOTE='/mnt/nfs2/engdes/wc296/paper6'
HOST='artemis-ood'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','-o','ControlPath=/private/tmp/hitac-artemis.sock',HOST]
RSYNC_SSH=shlex.join(SSH[:-1])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 statepath=ROOT/'data/runs/cost-field-full-launch-004.json';assert not statepath.exists()
 proof=ROOT/'data/provenance/cost-field-optimizer-local-verification-003.json'
 assert json.loads(proof.read_text())['state']=='passed_all_transferred32_codec_optimizer_records_and_independent_Adam_arithmetic'
 state=dict(state='running',pid=os.getpid(),phase='observe_existing_dataset_audit',started_unix=time.time(),optimizer_gate_sha256=sha(proof),stages=[])
 def save():
  temp=statepath.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(statepath)
 def stage(name,cmd):
  state['phase']=name;save();log=ROOT/'logs'/('cost-field-full-launch-004-'+name+'.log')
  with log.open('x') as f:r=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  state['stages'].append(dict(name=name,returncode=r.returncode,log=str(log.relative_to(ROOT)),sha256=sha(log)));save();assert r.returncode==0,name
  return log
 def remote_json(code):
  r=subprocess.run(SSH+['cd '+shlex.quote(REMOTE)+' && envs/torch-cu128-001/bin/python -c '+shlex.quote(code)],capture_output=True,text=True,timeout=45)
  assert r.returncode==0,r.stderr;return json.loads(r.stdout)
 def submit(name,script,absent_seeds):
  code='from pathlib import Path; import subprocess; '+ '; '.join('assert not Path('+repr('data/runs/cost-field-full-seed'+str(seed)+'-004')+').exists()' for seed in absent_seeds)+'; q=subprocess.check_output(["squeue","-h","-u","wc296","-o","%j"],text=True); assert '+repr(name)+' not in q; print("once_only_output_and_queue_check_passed")'
  stage('prelaunch_'+name,SSH+['cd '+shlex.quote(REMOTE)+' && envs/torch-cu128-001/bin/python -c '+shlex.quote(code)])
  # Persist intent before dispatch; an ambiguous outcome is not retried.
  state['submission_intent']=dict(job_name=name,script=script);save()
  log=stage('submit_'+name,SSH+['cd '+shlex.quote(REMOTE)+' && sbatch --parsable '+shlex.quote(script)])
  ids=[line for line in log.read_text().splitlines() if re.fullmatch(r'\d+(;\S+)?',line)]
  assert len(ids)==1,'Submission outcome requires actual queue/report discovery; do not resubmit'
  return ids[0].split(';')[0]
 save()
 try:
  code='''import json,hashlib,subprocess
from pathlib import Path
p=Path("data/provenance/cost-field-full-dataset-manifest-004.json")
if p.exists():
 r=json.loads(p.read_text()); assert r["state"]=="passed_whole_raw_KITTI_stereo_train3712_val3769_bytes_and_disjoint_splits" and r["total_files"]==29924
 print(json.dumps(dict(state="complete",sha256=hashlib.sha256(p.read_bytes()).hexdigest(),files=r["total_files"],bytes=r["total_bytes"])))
else:
 r=subprocess.run(["ps","-p","1980128","-o","pid,stat,args"],capture_output=True,text=True); assert r.returncode==0 and "cost-field-full-dataset-manifest-004.py" in r.stdout
 print(json.dumps(dict(state="existing_audit_running",actual_ps=r.stdout)))
'''
  failures=0
  while True:
   try:observed=remote_json(code);failures=0
   except (subprocess.TimeoutExpired,AssertionError,json.JSONDecodeError) as e:
    failures+=1;state.update(observation_error=repr(e),observation_failures=failures);save();assert failures<3;time.sleep(30);continue
   state['last_dataset_observation']=observed;save()
   if observed['state']=='complete':break
   time.sleep(30)
  rel='data/provenance/cost-field-full-dataset-manifest-004.json'
  stage('transfer_whole_dataset_manifest',['rsync','-az','-e',RSYNC_SSH,HOST+':'+REMOTE+'/'+rel,str(ROOT/'data/provenance')+'/'])
  assert sha(ROOT/rel)==observed['sha256']
  lock=json.loads((ROOT/'experiments/cost-field/GPU-inputs-003.json').read_text())
  files=['experiments/cost-field/protocol-004-full-training.md','experiments/cost-field/code_004/codec.py','experiments/cost-field/code_004/check_physical.py','experiments/cost-field/code_004/train_GPU.py','experiments/cost-field/GPU-full-17-004.sbatch','experiments/cost-field/GPU-full-23-41-004.sbatch','data/engineering/cost-field-physical-contract-004.json',rel]
  assert json.loads((ROOT/'data/engineering/cost-field-physical-contract-004.json').read_text())['state']=='passed_complete_physical_array_replay_contract'
  lock['frozen_inputs'].update({p:sha(ROOT/p) for p in files});lock.update(state='locked_before_full_KITTI_three_seed_task_training',dataset_manifest_path=rel,dataset_manifest_sha256=sha(ROOT/rel))
  target=ROOT/'experiments/cost-field/GPU-inputs-004.json';assert not target.exists();target.write_text(json.dumps(lock,indent=2)+'\n')
  stage('commit_full_input_lock',['git','add',str(target.relative_to(ROOT)),rel])
  stage('seal_full_input_lock',['git','commit','-m','research(inputs): whole29924 KITTI source files and full task training locked'])
  stage('stage_full_training',['rsync','-az','--relative','-e',RSYNC_SSH,*[p for p in files if p!=rel],str(target.relative_to(ROOT)),HOST+':'+REMOTE+'/'])
  job=submit('paper6-cost-full-17-004','experiments/cost-field/GPU-full-17-004.sbatch',[17]);state['seed17_job']=job;state['phase']='observe_actual_first_full_data_updates';save()
  code='''import json,subprocess
from pathlib import Path
p=Path("data/runs/cost-field-full-seed17-004/report.json")
r=json.loads(p.read_text()) if p.exists() else {}
q=subprocess.check_output(["squeue","-h","-u","wc296","-o","%i %T %j"],text=True)
print(json.dumps(dict(state=r.get("state"),updates=r.get("update_count",0),job_id=r.get("job_id"),last_update=r.get("last_update"),traceback=r.get("traceback"),queue=q)))
'''
  failures=0
  while True:
   try:r=remote_json(code);failures=0
   except (subprocess.TimeoutExpired,AssertionError,json.JSONDecodeError) as e:
    failures+=1;state['first_job_observation_error']=repr(e);save();assert failures<3;time.sleep(30);continue
   state['last_actual_full_training']=r;save();assert r['state']!='failed',r.get('traceback')
   if r['updates']>=32:
    assert str(r['job_id'])==job;break
   assert job in r['queue'],'Actual state needs inspection; do not restart'
   time.sleep(30)
  array=submit('paper6-cost-full-004','experiments/cost-field/GPU-full-23-41-004.sbatch',[23,41])
  state.update(state='finished_full_dataset_and_three_seed_training_dispatched_pending_full_closure',seed23_41_array_job=array,ended_unix=time.time());save()
  print(json.dumps({k:state[k] for k in ('state','seed17_job','seed23_41_array_job')}),flush=True)
 except BaseException as e:
  state.update(state='failed',error=repr(e),ended_unix=time.time());save();raise
if __name__=='__main__':main()
