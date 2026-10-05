"""One-shot launch gates outside frozen executable source trees; no weights tuned."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path('/home/sheng/paper6');sys.path.insert(0,str(root/'src'))
from geocomm.evidence import source_identity
p=argparse.ArgumentParser();p.add_argument('--engineering-result-commit',required=True);args=p.parse_args()
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
rid='stereo-native-task-seed17-001';sanity='stereo-native-task-sanity-001'
launch=root/'data/provenance'/f'{rid}-launch.json';prelaunch=root/'data/provenance'/f'{rid}-prelaunch.json'
if launch.exists() or prelaunch.exists() or (root/'data/runs'/f'{rid}.json').exists() or (Path('/mnt/d/paper6/runs')/rid).exists():
 raise ValueError('preserve prior outputs; never repeat an unknown launch')
source=root/'data/provenance'/f'server-source-manifest-{sanity}.json';saved=json.loads(source.read_text())
specs={'project':(root,['src','scripts','configs','pyproject.toml']),
 'liga':(root/'third_party/LIGA-Stereo',['liga','configs','tools','setup.py']),
 'mmdet':(root/'third_party/mmdetection_kitti',['mmdet']),
 'stereo_rcnn':(root/'third_party/Stereo-RCNN',['lib','demo.py','test_net.py'])}
for tree,spec in specs.items(): assert source_identity(*spec)==saved[tree]['actual_sources'],tree
run_path=root/'data/runs'/f'{sanity}.json';audit_path=root/'data/runs'/f'{sanity}-audit.json'
run=json.loads(run_path.read_text());audit=json.loads(audit_path.read_text())
assert run['state']=='finished' and run['optimizer_steps']==3 and run['frozen_states_identical']==519
assert audit['state']=='passed' and audit['steps']==audit['all_optimizer_update_counts']==3
assert audit['trained_parameter_tensors']==16 and audit['inactive_states_identical']==519
assert audit['manifest_sha256']==sha(run_path) and audit['source_manifest_sha256']==sha(source)
assert sha(run['checkpoint_path'])==audit['checkpoint_sha256']==run['checkpoint_sha256']
assert sha(run['full_initialization_path'])==run['full_initialization_sha256']==audit['full_initialization_sha256']
probe_path=root/'data/engineering/stereo-native-task-DDP-GPU-probe-001.json';probe=json.loads(probe_path.read_text())
assert probe['state']=='passed' and probe['required_states_identical']==535 and probe['checkpoint_sha256']==audit['checkpoint_sha256']
assert probe['project_sources']==saved['project']['actual_sources']
assert probe['checkpoint_loading']['state_hashes'] and len(probe['checkpoint_loading']['state_hashes'])==535
assert [r['condition'] for r in probe['conditions']]==['identity','awgn']
for condition in probe['conditions']:
 assert condition['calls']==dict(frames=1,student=2,codec=1,channel=1,build_cost=1,forbidden=0)
 row=condition['rows'][0];assert row['sequence']==['student','student','link_start','channel','link_done','build_cost']
 assert row['cost_inputs_are_received_features'] and row['appearance_input_is_received_feature']
 assert row['accounting']['total_complex_uses']==62400 and not row['autograd_enabled']
 assert row['left_stereo']['shape']==row['right_stereo']['shape']==[1,32,320,1248]
 assert row['appearance']['shape']==[1,32,80,312]
assert audit['summary']['nonzero_gradient_frames']=={k:3 for k in ('left_stereo','right_stereo','appearance','symbols','native_cost')}
assert run['all_modules_eval'] and not run['original_teacher_calls_allowed']
assert run['module_calls']==dict(steps=3,student=6,codec=3,channel=3,build_cost=3,map_to_bev=3,BEV=3,head3D=3,forbidden=0)
parent_closure=root/'data/provenance/stereo-feature-native-seed17-001-closure.json'
parent=json.loads(parent_closure.read_text());assert parent['state']=='closed_all_audits_passed'
assert parent['checkpoint_sha256']==run['initialization_sha256']=='9a4b291070df160e6c03e75ed31eeee1dda7f6bfbaa0528a43a9cdf526977553'
original_map_path=root/'data/provenance/cao2025-staged-trainer-source-bfee9d8.json'
original_map=json.loads(original_map_path.read_text());original_dir=root/'reproduction/cao2025'
assert len(original_map)==21 and all(sha(original_dir/name)==identity for name,identity in original_map.items())
assert sha(original_dir/'formal-protocol.md')=='682fb774e09b9be7f68d09f8251dac6d4b99af3290f02660951d398f9e00eace'
os.kill(26642,0)
# NVIDIA physical usage is the conservative coexistence gate, separate from
# WSL per-process CUDA mem_get_info reported by engineering scripts.
usage=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip().splitlines()
assert len(usage)==1
used,total=(int(v.strip()) for v in usage[0].split(','));DDP_peak=max(r['peak_reserved_bytes'] for r in probe['conditions']);peak=max(DDP_peak,int(run['peak_reserved_GiB']*2**30))
assert (total-used)*1024**2>peak+2*1024**3
formal_source=root/'data/provenance'/f'server-source-manifest-{rid}.json'
assert not formal_source.exists();formal_source.write_text(source.read_text())
protocol=root/'experiments/geometry-link/stereo-native-task-adaptation.md';script=root/'scripts/run_stereo_native_task_cycle.sh'
assert sha(protocol)==run['protocol_sha256']
record=dict(state='prelaunch_passed',run_id=rid,checked_at_unix=time.time(),implementation_commit='ad1541f',protocol_commit='2de22b1',
 engineering_result_commit=args.engineering_result_commit,engineering_manifest_sha256=sha(run_path),engineering_audit_sha256=sha(audit_path),
 engineering_probe_sha256=sha(probe_path),source_manifest_sha256=sha(formal_source),original21_source_states_unchanged=True,
 original_process_pid=26642,gpu_used_before_MiB=used,gpu_total_MiB=total,measured_DDP_peak_reserved_bytes=DDP_peak,measured_task_backward_peak_reserved_bytes=peak,
 script_sha256=sha(script),protocol_sha256=sha(protocol),initialization_sha256=run['initialization_sha256'],
 new_initialization='all535 F5b final states with freshAdamW; engineering weights excluded',planned_updates=3340,
 predetermined_final_evaluations=['identity_full372','awgn10dB_full372'],complex_uses=62400,
 scope='clean native3D codec task adaptation, no completed-training/AP claim')
prelaunch.write_text(json.dumps(record,indent=2)+'\n')
log=(root/'logs'/f'{rid}-cycle.log').open('x')
process=subprocess.Popen(['bash',str(script),rid],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
log.close();record.update(state='launched',pid=process.pid,started_at_unix=time.time());launch.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
