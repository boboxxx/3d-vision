"""Prelocked64 training-only observations, reusing sealed native engineering helpers."""
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import traceback

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'code'))
import probe_native as base
np=base.np;torch=base.torch;dist=base.dist
PREFIX='geometry-risk-pilot-001'
PROTOCOL=HERE.parent/'pilot-execution-001.md'


def sources():
    out=base.sources();out['pilot']=base.source_identity(ROOT,['experiments/geometry-risk/pilot-code'])
    return out


def metadata():
    out=base.metadata();out[str(PROTOCOL.relative_to(ROOT))]=base.sha256(PROTOCOL)
    return out


def require_engineering():
    manifest=ROOT/'data/engineering/geometry-risk-native-engineering-001.json'
    auditpath=ROOT/'data/provenance/geometry-risk-native-engineering-001-audit.json'
    run=base.read(manifest);audit=base.read(auditpath)
    assert run['state']=='finished_native_engineering_audit_pending'
    assert audit['state']=='passed_independent_native_engineering_audit' and audit['native_rows']==260
    assert audit['manifest_sha256']==base.sha256(manifest)
    assert audit['verifier_sha256']==base.sha256(ROOT/'data/provenance/verify-geometry-risk-native-engineering-001.py')
    assert audit['actual_terminal_PIDs']==[run['pid'],68491] and all(base.terminal(p) for p in audit['actual_terminal_PIDs'])
    assert run['source_identities']==run['source_identities_after']==base.sources()
    assert run['readonly_states']==535 and run['optimizer_updates']==0 and run['no_parameter_gradients']
    for path,h in audit['artifacts_sha256'].items():assert base.sha256(ROOT/path)==h
    # Replay discarded engineering noise exposure only, never import its damage targets.
    rng=np.random.Generator(np.random.PCG64(2801))
    for _ in range(258):rng.standard_normal((62400,2))
    assert rng.bit_generator.state==run['final_PCG64_state']
    return run,dict(manifest_sha256=base.sha256(manifest),audit_sha256=base.sha256(auditpath),
                    continued_PCG64_state=copy.deepcopy(rng.bit_generator.state),actual_terminal_PIDs=audit['actual_terminal_PIDs'])


def main():
    path=ROOT/'data/runs'/f'{PREFIX}.json';directory=Path('/mnt/d/paper6/runs')/PREFIX
    assert not path.exists() and not directory.exists(), 'preserve previous complete or failed observation run'
    lock=base.read(base.INPUT);expected=sources();meta=metadata();cpu=base.cpu_gate()
    engineering,eligibility=require_engineering()
    assert base.validate_previous(lock)==base.sources()
    selected=lock['future_fit_ids']+lock['future_out_of_fit_ids']
    assert len(selected)==len(set(selected))==64 and len(lock['future_fit_ids'])==32
    fold=base.read(ROOT/'data/internal-tuning-fold-001.json')
    train=sorted(fold['folds']['geocomm_tune_train']['ids'])
    assert selected==train[:64] and not set(selected)&set(fold['folds']['geocomm_tune_holdout']['ids'])
    assert base.sha256(ROOT/'data/internal-tuning-fold-001.json')==lock['fold_sha256']
    assert base.sha256(ROOT/'configs/tuning/stereo_task_seed17_epoch1.yaml')==lock['config_sha256']
    checkpoint=Path('/mnt/d/paper6/runs/stereo-native-task-seed17-002/checkpoint_epoch_1.pth')
    assert base.sha256(checkpoint)==lock['sole_parent_sha256']==engineering['sole_parent_sha256']
    used,total=map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(','))
    free=(total-used)*2**20;assert free>=12*2**30, 'physical GPU margin insufficient; no work started'
    directory.mkdir(parents=True)
    report=dict(state='starting',prefix=PREFIX,pid=os.getpid(),started_unix=time.time(),output_directory=str(directory),
                evidence_type='exploratory_training_only_native_risk_observations_no_fit_AP',source_identities=expected,
                metadata_identities=meta,CPU_gate_sha256=cpu,protocol_sha256=base.sha256(PROTOCOL),eligibility=eligibility,
                initial_PCG64_state=eligibility['continued_PCG64_state'],sole_parent_sha256=lock['sole_parent_sha256'],
                fit_ids=lock['future_fit_ids'],out_of_fit_ids=lock['future_out_of_fit_ids'],optimizer_updates=0,
                physical_free_before_bytes=free,frames={})
    base.save(path,report);group=observer=channel=None
    try:
        random.seed(17);np.random.seed(17);torch.manual_seed(17);torch.cuda.manual_seed_all(17);torch.backends.cudnn.benchmark=False
        os.chdir(base.LIGA)
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
        assert not dataset.training and dataset.split=='geocomm_tune_train'
        report['dataset']=base.fold_identity(dataset,fold)
        ids=[str(info['point_cloud']['lidar_idx']) for info in dataset.kitti_infos]
        group=tempfile.TemporaryDirectory(prefix='geometry-risk-pilot-')
        dist.init_process_group('nccl',init_method='file://'+group.name+'/rank',rank=0,world_size=1)
        model=build_network(config.MODEL,len(config.CLASS_NAMES),dataset).cuda().eval()
        reference=torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state']
        assert len(reference)==len(model.state_dict())==535;model.load_state_dict(reference,strict=True)
        for p in model.parameters():p.requires_grad_(False)
        assert base.assert_frozen(reference,model.state_dict(),set())==535
        observer=base.RiskObserver(model)
        channel=base.RiskChannel(model.backbone_3d.stereo_feature_link,directory/'noise',[(1,2,20,1248),(1,2,20,1248),(1,8,20,156)])
        channel.rng.bit_generator.state=copy.deepcopy(eligibility['continued_PCG64_state'])
        assert channel.calls==channel.prepared==0 and channel.rng.bit_generator.state==report['initial_PCG64_state']
        report.update(state='running',torch_version=torch.__version__,cuda_runtime=torch.version.cuda,native_layouts=channel.layouts)
        base.save(path,report)
        with (directory/'native-loss.jsonl').open('x') as stream:
            for frame in selected:
                assert sources()==expected and metadata()==meta and base.cpu_gate()==cpu
                result=base.measure_frame(frame,dataset.collate_batch([dataset[ids.index(frame)]]),model,observer,channel,directory,stream,reference)
                report['frames'][frame]=dict(report_path=str(directory/(frame+'.json')),report_sha256=base.sha256(directory/(frame+'.json')),
                    clean_loss=result['clean_loss'],full_AWGN10_loss=result['full_AWGN10_loss'],
                    source_files={str(dataset.root_path.resolve()/'training'/component/(frame+suffix)):base.sha256(dataset.root_path.resolve()/'training'/component/(frame+suffix))
                                  for component,suffix in [('image_2','.png'),('image_3','.png'),('label_2','.txt'),('calib','.txt')]})
                assert sources()==expected and metadata()==meta and base.cpu_gate()==cpu
                assert torch.cuda.max_memory_reserved()+2*2**30<=free
                report.update(completed_frames=len(report['frames']),completed_native_passes=channel.calls,checked_unix=time.time())
                base.save(path,report)
        calls=dict(steps=8320,student=16640,codec=8320,channel=8320,build_cost=8320,map_to_bev=8320,BEV=8320,head3D=8320,forbidden=0)
        assert observer.calls==calls and channel.calls==channel.prepared==8320 and channel.pending is None
        report.update(state='finished64_native_observations_independent_audit_pending',calls=calls,channel_attempts=8320,
                      attempted_complex_uses=8320*62400,readonly_states=535,all535_final_state_hashes=base.state_hashes(model),
                      no_parameter_gradients=all(p.grad is None for p in model.parameters()),records_sha256=base.sha256(directory/'native-loss.jsonl'),
                      final_PCG64_state=channel.rng.bit_generator.state,source_identities_after=sources(),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        assert report['source_identities_after']==expected and metadata()==meta
    except BaseException:
        report.update(state='failed',traceback=traceback.format_exc());raise
    finally:
        if channel is not None:channel.close()
        if observer is not None:observer.close()
        if dist.is_initialized():dist.destroy_process_group()
        if group is not None:group.cleanup()
        report['ended_unix']=time.time();base.save(path,report)
        print(json.dumps({'state':report['state'],'completed_frames':len(report['frames'])}),flush=True)


if __name__=='__main__':main()
