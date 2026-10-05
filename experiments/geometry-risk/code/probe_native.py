"""Two-frame frozen native loss intervention engineering; not the full fitted pilot."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
import torch
import torch.distributed as dist

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];LIGA=ROOT/'third_party/LIGA-Stereo'
sys.path[:0]=[str(HERE),str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'experiments/geometry-link/F7/code'),str(LIGA),str(ROOT/'third_party/mmdetection_kitti')]
from geocomm.evidence import sha256,source_identity
from geocomm.stereo_task_adaptation import NativeTaskAdaptation,SEQUENCE,assert_frozen
from geocomm.inference import SENSOR_INPUT_KEYS
from conditions import augmented_evidence
from pretrain_student import fold_identity
from native_transport import RiskChannel,array_identity,aggregate_geometry,cell_map
from proxy import epipolar_proxy

INPUT=HERE.parent/'native-engineering-inputs-001.json'


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.part');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def terminal(pid):
    try:os.kill(pid,0)
    except ProcessLookupError:return True
    return False


def source_specs():
    return dict(project=(ROOT,['src','scripts','configs','pyproject.toml']),
                liga=(LIGA,['liga','configs','tools','setup.py']),
                mmdet=(ROOT/'third_party/mmdetection_kitti',['mmdet']),
                stereo_rcnn=(ROOT/'third_party/Stereo-RCNN',['lib','demo.py','test_net.py']),
                F7=(ROOT,['experiments/geometry-link/F7/code']),
                final_evaluation=(ROOT,['experiments/original-detector-integration/final-code','experiments/original-detector-integration/adapter.py']),
                experiment=(ROOT,['experiments/geometry-risk/code']))


def sources():return {k:source_identity(*v) for k,v in source_specs().items()}


def metadata():
    return {str(p.relative_to(ROOT)):sha256(p) for p in (INPUT,HERE.parent/'protocol-001.md',HERE.parent/'native-engineering-001.md')}


def cpu_gate():
    path=ROOT/'data/engineering/geometry-risk-native-sheng-CPU-001.json'
    result=read(path)
    assert result['state']=='passed' and result['tests']==8 and result['CUDA_visible_devices']==''
    assert result['torch']=='2.5.0+cu118'
    for name,value in result['source_sha256'].items():assert sha256(ROOT/name)==value
    return sha256(path)


def validate_previous(lock):
    assert all(terminal(pid) for pid in lock['wait_PIDs']), 'previous physical processes still live'
    current=sources()
    for prefix in lock['required_closure_prefixes']:
        closure=read(ROOT/'data/provenance'/f'{prefix}-closure.json')
        assert closure['state']=='closed_all_audits_passed'
        for name,value in closure['artifacts_sha256'].items():assert sha256(ROOT/name)==value
        identities=closure.get('source_sha256')
        if identities:
            for key,value in identities.items():assert current['F7' if key=='experiment' else key]['sha256']==value
        else:
            for key,value in closure['sources'].items():assert current[key]==value
    return current


def state_hashes(model):return {name:array_identity(value) for name,value in model.state_dict().items()}


class RiskObserver(NativeTaskAdaptation):
    def __init__(self,model):
        super().__init__(model);self.capture_clean=False;self.clean_features=[]

    def student(self,module,inputs,outputs):
        super().student(module,inputs,outputs)
        if self.capture_clean:
            self.clean_features.append(tuple(value.detach().cpu().numpy().copy() for value in outputs))

    def finish_forward(self):
        if self.pending is None or self.pending['sequence']!=SEQUENCE or self.pending['gradient_norms']:
            raise RuntimeError('incomplete/accidentally backpropagated intervention forward')
        if set(self.pending['gradient_shapes'])!={'symbols','native_cost','left_stereo','right_stereo','appearance'}:
            raise RuntimeError('missing connected native receiver graph')
        self.calls['steps']+=1;row=self.pending;self.pending=None;self.received=None;self.gt_boxes=None
        self.model.dense_head.forward_ret_dict.clear()
        for handle in self.gradient_handles:handle.remove()
        self.gradient_handles=[]
        return row


def descriptor_energy(features):
    values=[]
    for value in features:
        energy=np.square(value[0].astype(np.float64)).mean(0);ids=cell_map(*energy.shape)
        sums=np.bincount(ids.ravel(),weights=energy.ravel(),minlength=32)
        counts=np.bincount(ids.ravel(),minlength=32)
        values.append(sums/counts)
    return np.mean(values,axis=0)


def measure_frame(frame,batch,model,observer,channel,directory,stream,reference):
    sensors={k:batch[k] for k in SENSOR_INPUT_KEYS}
    for k in ('left_img','right_img'):sensors[k]=torch.as_tensor(sensors[k],device='cuda',dtype=torch.float32)
    targets=torch.as_tensor(batch['gt_boxes'],device='cuda',dtype=torch.float32)
    assert tuple(sensors['left_img'].shape)==tuple(sensors['right_img'].shape)==(1,3,320,1248)
    assert str(sensors['frame_id'][0])==frame and not sensors['calib'][0].flipped and 'random_T' not in batch
    fingerprint=augmented_evidence(sensors,targets)
    fb=float(sensors['calib'][0].fu_mul_baseline);assert fb>0
    hashes_before=state_hashes(model);assert len(hashes_before)==535
    channel.begin_frame();observer.capture_clean=True;observer.clean_features=[]
    channel.prepare(frame,'clean')
    loss=observer.forward(sensors,targets);clean=float(loss.detach());symbols=channel.leaf.detach().cpu().numpy()[0].copy()
    gradient,=torch.autograd.grad(loss,channel.leaf)
    assert gradient.shape==(1,62400,2) and torch.isfinite(gradient).all()
    gradient=gradient.detach().cpu().numpy()[0].copy();row=observer.finish_backward()
    condition=channel.finish();row.update(loss=clean,condition=condition,augmented_evidence=fingerprint)
    stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
    features=(observer.clean_features[0][0],observer.clean_features[1][0],observer.clean_features[0][1])
    assert [list(f.shape) for f in features]==[[1,32,320,1248],[1,32,320,1248],[1,32,80,312]]
    observer.capture_clean=False;observer.clean_features=[];del loss
    valid_shape=(min(int(sensors['image_shape'][0][0]),320),min(int(sensors['image_shape'][0][1]),1280))
    result=epipolar_proxy(features[0][0],features[1][0],focal_baseline=fb,valid_image_shape=valid_shape)
    scores=aggregate_geometry(result,symbols,gradient,channel.group_ids)
    native_energy=descriptor_energy(features)
    for score,energy in zip(scores,native_energy):score['native_descriptor_energy_mean']=float(energy)
    arrays=dict(left_stereo=features[0],right_stereo=features[1],appearance=features[2],symbols=symbols,gradient=gradient,
                left_img=sensors['left_img'].detach().cpu().numpy(),right_img=sensors['right_img'].detach().cpu().numpy(),
                gt_boxes=targets.detach().cpu().numpy(),P2=sensors['calib'][0].P2,P3=sensors['calib'][0].P3,
                image_shape=np.asarray(sensors['image_shape']),valid_image_shape=np.asarray(valid_shape),group_ids=channel.group_ids,disparity_bins=result['disparity_bins_input_pixels'])
    for name,value in vars(sensors['calib'][0]).items():arrays['calib_'+name]=np.asarray(value)
    for view in ('left','right'):
        for name,value in result[view].items():arrays[view+'_'+name]=value
    fixture=directory/f'{frame}-clean-fixture.npz';np.savez_compressed(fixture,**arrays)
    del arrays,features,result,symbols,gradient
    changes=[];full_loss=None
    for group in range(32):
        deltas=[]
        for draw in range(4):
            channel.prepare(frame,'group',group,draw);loss=observer.forward(sensors,targets)
            value=float(loss.detach());record=observer.finish_forward();condition=channel.finish()
            assert all(p.grad is None and not p.requires_grad for p in model.parameters())
            record.update(loss=value,loss_minus_clean=value-clean,condition=condition,augmented_evidence_sha256=fingerprint['sha256'])
            stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();deltas.append(value-clean);del loss
        changes.append(dict(group=group,signed_damage_mean=float(np.mean(deltas)),positive_damage_mean=max(float(np.mean(deltas)),0.),
                            sample_variance=float(np.var(deltas,ddof=1)),draw_damage=deltas))
    channel.prepare(frame,'full_awgn');loss=observer.forward(sensors,targets);full_loss=float(loss.detach())
    record=observer.finish_forward();condition=channel.finish()
    record.update(loss=full_loss,loss_minus_clean=full_loss-clean,condition=condition,augmented_evidence_sha256=fingerprint['sha256'])
    stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();del loss
    assert augmented_evidence(sensors,targets)==fingerprint
    assert assert_frozen(reference,model.state_dict(),set())==535 and state_hashes(model)==hashes_before
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    report=dict(frame_id=frame,clean_loss=clean,full_AWGN10_loss=full_loss,focal_baseline=fb,valid_image_shape=list(valid_shape),scores=scores,damage=changes,
                fixture_path=str(fixture),fixture_sha256=sha256(fixture),fixture_bytes=fixture.stat().st_size,
                augmented_evidence=fingerprint,state_hashes_before=hashes_before,state_hashes_after=state_hashes(model),
                native_loss_passes=130,attempted_complex_uses=130*62400,no_parameter_gradients=True)
    save(directory/f'{frame}.json',report);return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prefix',required=True);args=parser.parse_args()
    assert args.prefix=='geometry-risk-native-engineering-001', 'only the locked engineering prefix is authorized'
    manifest=ROOT/'data/engineering'/f'{args.prefix}.json';directory=Path('/mnt/d/paper6/runs')/args.prefix
    assert not manifest.exists() and not directory.exists(), 'preserve earlier attempt'
    lock=read(INPUT);expected=sources();meta=metadata();cpu_sha=cpu_gate();report=dict(state='starting',prefix=args.prefix,pid=os.getpid(),started_unix=time.time(),
        evidence_type='engineering_only_no_fit_or_AP',input_sha256=sha256(INPUT),protocol_sha256=sha256(HERE.parent/'native-engineering-001.md'),
        source_identities=expected,metadata_identities=meta,CPU_gate_sha256=cpu_sha,output_directory=str(directory),optimizer_updates=0)
    group=None;observer=None;channel=None
    directory.mkdir(parents=True);save(manifest,report)
    try:
        assert expected==validate_previous(lock)
        assert sha256(ROOT/'data/internal-tuning-fold-001.json')==lock['fold_sha256']
        assert sha256(ROOT/'configs/tuning/stereo_task_seed17_epoch1.yaml')==lock['config_sha256']
        assert sha256(HERE.parent/'protocol-001.md')==lock['protocol_sha256']
        checkpoint=Path('/mnt/d/paper6/runs/stereo-native-task-seed17-002/checkpoint_epoch_1.pth')
        assert sha256(checkpoint)==lock['sole_parent_sha256']
        used,total=map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(','))
        free=(total-used)*2**20;assert free>=12*2**30
        random.seed(17);np.random.seed(17);torch.manual_seed(17);torch.cuda.manual_seed_all(17);torch.backends.cudnn.benchmark=False
        os.chdir(LIGA)
        from easydict import EasyDict
        from liga.config import cfg_from_yaml_file
        from liga.datasets import build_dataloader
        from liga.models import build_network
        from liga.utils.common_utils import create_logger
        config=cfg_from_yaml_file(str(ROOT/'configs/tuning/stereo_task_seed17_epoch1.yaml'),EasyDict())
        config.DATA_CONFIG.DATA_SPLIT.test='geocomm_tune_train'
        config.DATA_CONFIG.INFO_PATH.test=['kitti_infos_geocomm_tune_train.pkl']
        assert not config.DATA_CONFIG.FORCE_FLIP and len(config.DATA_CONFIG.TEST_DATA_AUGMENTOR)==1
        dataset,_,_=build_dataloader(config.DATA_CONFIG,config.CLASS_NAMES,batch_size=1,dist=False,workers=0,training=False,logger=create_logger(directory/'dataset.log'))
        report['dataset']=fold_identity(dataset,read(ROOT/'data/internal-tuning-fold-001.json'))
        assert not dataset.training and dataset.split=='geocomm_tune_train'
        ids=[str(info['point_cloud']['lidar_idx']) for info in dataset.kitti_infos]
        indices=[ids.index(frame) for frame in lock['engineering_ids']]
        group=tempfile.TemporaryDirectory(prefix='geometry-risk-native-');dist.init_process_group('nccl',init_method='file://'+group.name+'/rank',rank=0,world_size=1)
        model=build_network(config.MODEL,len(config.CLASS_NAMES),dataset).cuda().eval()
        reference=torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state']
        assert len(reference)==len(model.state_dict())==535;model.load_state_dict(reference,strict=True)
        for parameter in model.parameters():parameter.requires_grad_(False)
        assert assert_frozen(reference,model.state_dict(),set())==535
        observer=RiskObserver(model)
        channel=RiskChannel(model.backbone_3d.stereo_feature_link,directory/'noise',[(1,2,20,1248),(1,2,20,1248),(1,8,20,156)])
        assert len(channel.group_ids)==62400 and set(channel.group_ids)==set(range(32))
        report.update(state='running',torch_version=torch.__version__,cuda_runtime=torch.version.cuda,
                      native_layouts=channel.layouts,physical_free_before_bytes=free,sole_parent_sha256=sha256(checkpoint),frames={})
        save(manifest,report)
        with (directory/'native-loss.jsonl').open('x') as stream:
            for frame,index in zip(lock['engineering_ids'],indices):
                batch=dataset.collate_batch([dataset[index]])
                result=measure_frame(frame,batch,model,observer,channel,directory,stream,reference)
                report['frames'][frame]=dict(report_path=str(directory/f'{frame}.json'),report_sha256=sha256(directory/f'{frame}.json'),
                                             clean_loss=result['clean_loss'],full_AWGN10_loss=result['full_AWGN10_loss'],
                                             source_files={str(dataset.root_path.resolve()/'training'/component/(frame+suffix)):sha256(dataset.root_path.resolve()/'training'/component/(frame+suffix))
                                                           for component,suffix in [('image_2','.png'),('image_3','.png'),('label_2','.txt'),('calib','.txt')]})
                assert sources()==expected and metadata()==meta and cpu_gate()==cpu_sha
                assert torch.cuda.max_memory_reserved()+2*2**30<=free
                save(manifest,report)
        assert observer.calls==dict(steps=260,student=520,codec=260,channel=260,build_cost=260,map_to_bev=260,BEV=260,head3D=260,forbidden=0)
        assert channel.calls==channel.prepared==260 and channel.pending is None
        report.update(state='finished_native_engineering_audit_pending',calls=observer.calls,channel_attempts=260,
                      attempted_complex_uses=260*62400,readonly_states=535,all535_final_state_hashes=state_hashes(model),
                      no_parameter_gradients=all(p.grad is None for p in model.parameters()),
                      records_sha256=sha256(directory/'native-loss.jsonl'),final_PCG64_state=channel.rng.bit_generator.state,
                      source_identities_after=sources(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        assert report['source_identities_after']==expected and metadata()==meta
    except BaseException:
        report.update(state='failed',traceback=traceback.format_exc());raise
    finally:
        if channel is not None:channel.close()
        if observer is not None:observer.close()
        if dist.is_initialized():dist.destroy_process_group()
        if group is not None:group.cleanup()
        report['ended_unix']=time.time();save(manifest,report);print(json.dumps({'state':report['state'],'prefix':args.prefix}),flush=True)


if __name__=='__main__':main()
