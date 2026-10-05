"""Unique sequential 12-native-endpoint/12-audit cycle and terminal closure."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/home/sheng/paper6')
CODE='experiments/original-paper-scenarios/source-cache-inference-code'
sys.path.insert(0,str(ROOT/CODE))
import cache_contract as c


def checks(scope):
    local=c.read(ROOT/'data/engineering/source-cache-inference-local-CPU-001.json')
    native=c.read(ROOT/'data/engineering/source-cache-inference-sheng-CPU-001.json')
    assert local['state']==native['state']=='passed' and local['tests']==native['tests']==5
    assert local['sources']==native['sources']==c.sources()
    c.proof(scope)
    if scope=='main':
        proof=c.read(ROOT/'data/provenance/source-cache-inference-engineering-001-local-verification.json')
        assert proof['state']=='passed_all24_transferred_native_inference_records_and_predictions'
        assert proof['sources']==c.sources() and proof['frames']==24
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    return local['sources']


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('launch','run','close'),required=True)
    p.add_argument('--scope',choices=('engineering','main'),required=True);args=p.parse_args()
    prefix='source-cache-inference-'+args.scope+'-001'
    run=ROOT/'data/runs'/(prefix+'-cycle.json');launch=ROOT/'data/runs'/(prefix+'-launch.json')
    closure=ROOT/'data/provenance'/(prefix+'-closure.json');script=Path(__file__).resolve()
    if args.mode=='launch':
        sources=checks(args.scope);log=ROOT/'logs'/(prefix+'-controller.log')
        assert not any(x.exists() for x in (run,launch,closure,log))
        for condition in c.CONDITIONS:
            for detector in ('stereo_rcnn','liga'):
                name=prefix+'-'+condition+'-'+detector
                assert not (ROOT/'data/runs'/(name+'.json')).exists() and not (c.NATIVE/name).exists()
        command=[sys.executable,str(script),'--mode','run','--scope',args.scope]
        with log.open('x') as stream:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        value=dict(state='launched_once',prefix=prefix,scope=args.scope,pid=child.pid,launched_unix=time.time(),
                   command=command,controller_sha256=c.sha(script),sources=sources,log=str(log))
        with launch.open('x') as stream:json.dump(value,stream,indent=2)
        print(json.dumps(value));return
    if args.mode=='run':
        sources=checks(args.scope);assert not run.exists()
        value=dict(state='running',prefix=prefix,scope=args.scope,pid=os.getpid(),sources=sources,
                   controller_sha256=c.sha(script),started_unix=time.time(),completed_commands=[],endpoints={})
        c.save(run,value)
        try:
            for condition in c.CONDITIONS:
                for detector in ('stereo_rcnn','liga'):
                    name=prefix+'-'+condition+'-'+detector;manifest=ROOT/'data/runs'/(name+'.json')
                    audit=ROOT/'data/provenance'/(name+'-audit.json')
                    commands=[[sys.executable,CODE+'/evaluate.py','--scope',args.scope,'--condition',condition,'--detector',detector],
                              [sys.executable,CODE+'/audit.py','--manifest',str(manifest),'--output',str(audit)]]
                    for command in commands:
                        value.update(current_condition=condition,current_detector=detector);c.save(run,value)
                        index=len(value['completed_commands']);log=ROOT/'logs'/(prefix+'-command-'+str(index)+'.log')
                        started=time.time()
                        with log.open('x') as stream:
                            child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT);status=child.wait()
                        row=dict(command=command,pid=child.pid,started_unix=started,ended_unix=time.time(),returncode=status,
                                 log=str(log),log_sha256=c.sha(log));value['completed_commands'].append(row);c.save(run,value)
                        assert status==0,row
                    report=c.read(audit);assert report['state']=='passed' and report['frames']==(2 if args.scope=='engineering' else 3769)
                    value['endpoints'][condition+'-'+detector]=dict(manifest=str(manifest),manifest_sha256=c.sha(manifest),
                                                                  audit=str(audit),audit_sha256=c.sha(audit))
                    c.save(run,value);assert c.sources()==sources
            assert len(value['endpoints'])==12 and len(value['completed_commands'])==24
            value.update(state='finished12_native_endpoints12_audits_pending_terminal_closure',ended_unix=time.time())
        except BaseException as error:
            value.update(state='failed',error=repr(error),ended_unix=time.time());raise
        finally:c.save(run,value)
        return
    assert not closure.exists();start=c.read(launch);cycle=c.read(run)
    ps=subprocess.run(['ps','-p',str(start['pid']),'-o','pid,stat,args'],capture_output=True,text=True)
    assert ps.returncode==1 and cycle['pid']==start['pid']
    assert cycle['state']=='finished12_native_endpoints12_audits_pending_terminal_closure'
    assert start['sources']==cycle['sources']==c.sources()
    assert start['controller_sha256']==cycle['controller_sha256']==c.sha(script)
    assert len(cycle['completed_commands'])==24 and all(x['returncode']==0 for x in cycle['completed_commands'])
    artifacts={launch,run,Path(start['log'])};frames=0;shared={};endpoints={}
    for command in cycle['completed_commands']:
        path=Path(command['log']);assert c.sha(path)==command['log_sha256'];artifacts.add(path)
    for label,item in cycle['endpoints'].items():
        manifest=Path(item['manifest']);audit=Path(item['audit'])
        assert c.sha(manifest)==item['manifest_sha256'] and c.sha(audit)==item['audit_sha256']
        record=c.read(manifest);report=c.read(audit)
        assert record['state']=='finished' and report['state']=='passed' and report['manifest_sha256']==c.sha(manifest)
        assert record['sources']==report['sources']==c.sources()
        assert report['frames']==len(record['ordered_ids'])==len(record['frames'])==(2 if args.scope=='engineering' else 3769)
        assert record['detector_initial_state_hashes']==record['detector_final_state_hashes']
        directory=Path(record['output_dir']);artifacts.update((manifest,audit))
        for row in record['frames']:
            frame=row['frame_id'];prediction=directory/'data'/(frame+'.txt')
            assert c.sha(prediction)==row['prediction_sha256']==report['files'][frame]['prediction_sha256']
            assert c.sha(row['cache_path'])==row['cache_sha256']==report['files'][frame]['cache_sha256']
            calib=c.DATA/'training/calib'/(frame+'.txt');assert c.sha(calib)==row['calibration_sha256']
            identity=dict(cache_sha256=row['cache_sha256'],received_arrays=row['received_arrays'],received_tensors=row['received_tensors'])
            key=(record['condition'],frame)
            assert key not in shared or shared[key]==identity;shared[key]=identity
            artifacts.add(prediction);frames+=1
            if args.scope=='main':assert c.sha(c.DATA/'training/label_2'/(frame+'.txt'))==row['label_sha256']
        if args.scope=='main':
            assert c.sha(directory/'metrics.json')==record['metrics_sha256'] and report['recomputed_metrics']==record['metrics']
            artifacts.update((directory/'metrics.json',directory/'evaluator.txt'))
        endpoints[label]=dict(manifest_sha256=c.sha(manifest),audit_sha256=c.sha(audit),frames=report['frames'],
                              metrics=record['metrics'],states=record['detector_states'])
    assert frames==(24 if args.scope=='engineering' else 45228) and len(shared)==frames//2
    def mapped(path):
        if path.is_relative_to(ROOT):return str(path.relative_to(ROOT))
        assert path.is_relative_to('/mnt/d/paper6/runs')
        return 'data/engineering/'+prefix+'-native-transfer/'+str(path).lstrip('/')
    result=dict(state='closed_all12_endpoints_actual_terminal_and_shared_received_inputs',scope=args.scope,actual_terminal=True,
                checked_unix=time.time(),pid=start['pid'],frames=frames,shared_pairs=len(shared),sources=c.sources(),endpoints=endpoints,
                artifacts_sha256={mapped(p):c.sha(p) for p in sorted(artifacts)},
                native_to_local_artifacts={mapped(p):str(p) for p in sorted(artifacts)})
    with closure.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    (ROOT/'data/provenance'/(prefix+'-transfer-files.txt')).write_text('\n'.join(str(p.relative_to(ROOT)) for p in sorted(artifacts) if p.is_relative_to(ROOT))+'\n')
    (ROOT/'data/provenance'/(prefix+'-native-transfer-files.txt')).write_text('\n'.join(str(p).lstrip('/') for p in sorted(artifacts) if not p.is_relative_to(ROOT))+'\n')
    print(json.dumps(dict(state=result['state'],frames=frames,artifacts=len(artifacts))))


if __name__=='__main__':main()
