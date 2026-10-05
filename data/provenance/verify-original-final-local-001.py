"""Complete transferred artifacts/predictions audit. Does not rerun AP or 28GB RGB."""
import hashlib,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
PREFIX='original-final-native-seed17-001'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())


def resources(roi):
    h,w=roi['views'][0]['shape'];hp=h+(-h)%6;wp=w+(-w)%6;cells=[]
    for view in roi['views']:
        mask=np.zeros((hp//2,wp//2),bool)
        for box in view['boxes']:
            x1,y1,x2,y2=box['xyxy'];mask[y1//2:(y2+1)//2,x1//2:(x2+1)//2]=True
        cells.append(int(mask.sum()))
    real=9*(2*(hp//6)*(wp//6)+sum(cells));control=1176+672*sum(len(v['boxes']) for v in roi['views'])
    return dict(key_cells=cells,data_real_values=real,padding_real_values=real%2,data_uses=(real+1)//2,
                control_uses=control,pilot_uses=0,total_uses=(real+1)//2+control,
                cbr_complex_per_rgb_real_value=((real+1)//2+control)/(6*h*w))


def main():
    cp=ROOT/'data/provenance'/f'{PREFIX}-closure.json';closure=read(cp)
    terminalpath=ROOT/'data/provenance'/f'{PREFIX}-actual-terminal-001.json';terminal=read(terminalpath)
    assert closure['state']=='closed_all_audits_passed'
    assert terminal['cycle_pid']==56656 and terminal['wrapper_pid']==56638
    assert terminal['cycle_actual_terminal'] and terminal['wrapper_actual_terminal']
    assert terminal['closure_sha256']==sha(cp) and terminal['artifacts_sha256']==closure['artifacts_sha256']
    assert terminal['source_identities']==closure['sources']
    for p,h in closure['artifacts_sha256'].items():assert sha(ROOT/p)==h
    for p,h in terminal['prediction_bundle_sha256'].items():assert sha(ROOT/p)==h
    assert len(terminal['prediction_bundle_sha256'])==2982
    roots={'project':ROOT,'liga':ROOT/'third_party/LIGA-Stereo','mmdet':ROOT/'third_party/mmdetection_kitti',
           'stereo_rcnn':ROOT/'third_party/Stereo-RCNN','final_evaluation':ROOT}
    for key,item in closure['sources'].items():
        files={p:sha(roots[key]/p) for p in item['file_hashes']};assert files==item['file_hashes']
        assert hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()==item['sha256']
    assert closure['final_chain']['checkpoint_sha256']=='08a7c848dd4f11b8d55badb87793e2ad4d3c5c009395b4134a593bee49daf674'
    fold=read(ROOT/'data/internal-tuning-fold-001.json');ids=fold['folds']['geocomm_tune_holdout']['ids']
    assert len(set(ids))==len(ids)==372
    roirows=[json.loads(line) for line in (ROOT/'data/engineering/cao2025-roi-holdout-001.jsonl').read_text().splitlines()]
    assert [r['frame_id'] for r in roirows]==ids
    caches={};cachemetrics={}
    for channel in ('identity','awgn'):
        path=ROOT/'data/runs'/f'{PREFIX}-cache-{channel}.json';c=read(path);audit=read(path.with_name(path.stem+'-audit.json'))
        assert c['state']=='finished' and audit['state']=='passed' and audit['manifest_sha256']==sha(path)
        assert c['channel']==channel and c['seed']==17 and c['snr_db']==10 and c['device']=='cpu'
        assert c['ordered_ids']==ids and c['frames_complete']==audit['frames']==372
        assert c['codec_states_readonly']==len(c['codec_initial_state_hashes'])==768
        assert c['codec_initial_state_hashes']==c['codec_final_state_hashes']
        assert c['source_identities']==closure['sources'] and c['final_chain']==closure['final_chain']
        uses=energy=erasures=0
        for frame,roi in zip(ids,roirows):
            row=c['frames'][frame];expected=resources(roi)
            assert all(row['accounting'][k]==v for k,v in expected.items())
            assert row['received_file_sha256']==audit['files'][frame]['received_file_sha256']
            assert row['received_tensors']==audit['files'][frame]['received_tensors']
            assert row['sensor_sha256']=={v['camera']:v['image_sha256'] for v in roi['views']}
            assert row['receiver_calls']==dict(receive_payload=1,semantic_decode=int(row['erasure'] is None))
            if channel=='identity':assert row['erasure'] is None and row['generator_before_state_sha256']==row['generator_after_state_sha256']
            uses+=expected['total_uses'];energy+=row['accounting']['total_energy'];erasures+=row['erasure'] is not None
        assert audit['attempted_complex_uses']==uses and audit['attempted_energy']==energy and audit['erasures']==erasures
        caches[channel]=c;cachemetrics[channel]=dict(attempted_complex_uses=uses,mean_complex_uses=uses/372,energy=energy,erasures=erasures)
    bundle=ROOT/'data/provenance'/f'{PREFIX}-prediction-bundle-001';runs={};metrics={}
    assert set(closure['endpoints'])=={d+':'+c for d in ('stereo_rcnn','liga') for c in ('clean_relay','identity','awgn')}
    for key,item in closure['endpoints'].items():
        name=Path(item['manifest_path']).name;runpath=ROOT/'data/runs'/name;run=read(runpath)
        auditpath=ROOT/'data/runs'/Path(item['audit_path']).name;audit=read(auditpath)
        assert sha(runpath)==item['manifest_sha256']==audit['manifest_sha256'] and sha(auditpath)==item['audit_sha256']
        assert run['state']=='finished' and audit['state']=='passed' and run['ordered_ids']==ids and run['frames_complete']==audit['frames']==372
        count=484 if run['detector']=='liga' else 670
        assert run['detector_states_readonly']==audit['detector_states_readonly']==len(run['detector_initial_state_hashes'])==count
        assert run['detector_initial_state_hashes']==run['detector_final_state_hashes']
        assert run['source_identities']==closure['sources'] and run['final_chain']==closure['final_chain']
        assert run['metrics']==audit['recomputed_metrics']==item['metrics']
        directory=bundle/key.replace(':','-');assert sha(directory/'metrics.json')==run['metrics_sha256'] and read(directory/'metrics.json')==run['metrics']
        assert {p.stem for p in directory.glob('*.txt')}==set(ids)
        predictions=erasures=uses=0
        for frame in ids:
            row=run['frames'][frame];fileaudit=audit['files'][frame];text=(directory/(frame+'.txt')).read_text()
            assert sha(directory/(frame+'.txt'))==row['prediction_sha256']==fileaudit['prediction_sha256']
            for folder,hkey in [('calib','calibration_sha256'),('label_2','label_sha256')]:
                assert sha(bundle/folder/(frame+'.txt'))==row[hkey]==fileaudit[hkey]
            lines=text.splitlines();assert len(lines)==row['prediction_count'];predictions+=len(lines)
            for line in lines:
                fields=line.split();assert len(fields)==16 and fields[0] in ('Car','Pedestrian','Cyclist')
                numbers=list(map(float,fields[1:]));assert all(map(math.isfinite,numbers)) and 0<=numbers[-1]<=1
            if row['erasure'] is not None:assert not lines and not row['detector_called'] and row['calls']=={};erasures+=1
            channel=run['channel']
            if channel=='clean_relay':assert row['accounting'] is None and row['erasure'] is None
            else:
                source=caches[channel]['frames'][frame]
                for field in ('received_file_sha256','received_tensors','erasure','accounting','sensor_sha256'):assert row[field]==source[field]
                uses+=row['accounting']['total_uses']
            if row['erasure'] is None:
                if run['detector']=='liga':assert row['calls']==dict(image_backbone=2,feature_neck=2,build_cost=1,head3D=1,forbidden=0)
                else:
                    assert row['GT_placeholder_only'] and row['native_3D_counts']['dense_solutions']==len(lines)
                    assert row['calls']==dict(detector=1,image_backbone=2,dense_alignment=int(row['native_3D_counts']['initial_solutions']>0))
        assert audit['predictions']==predictions and audit['erasures']==erasures and audit['attempted_complex_uses']==uses
        runs[key]=run;metrics[key]=run['metrics']
    for channel in ('clean_relay','identity','awgn'):
        a,b=[runs[d+':'+channel] for d in ('stereo_rcnn','liga')]
        assert a['cache_manifest_sha256']==b['cache_manifest_sha256']
        for frame in ids:
            for field in ('received_file_sha256','received_tensors','erasure','accounting','sensor_sha256'):assert a['frames'][frame][field]==b['frames'][frame][field]
    output=ROOT/'data/provenance'/f'{PREFIX}-local-verification-001.json'
    result=dict(state='passed_complete_transferred_metadata_predictions_GT_resources',closure_sha256=sha(cp),terminal_sha256=sha(terminalpath),
                verifier_sha256=sha(Path(__file__)),retained_artifacts=len(closure['artifacts_sha256']),prediction_GT_files=2982,
                caches=cachemetrics,metrics=metrics,scope='Complete local text/metadata/resource/source audit plus sealed fresh server AP and large-array audits. No local AP or 28GB RGB tensor reexecution.')
    with output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'state':result['state'],'retained_artifacts':result['retained_artifacts'],'prediction_GT_files':2982}))


if __name__=='__main__':main()
