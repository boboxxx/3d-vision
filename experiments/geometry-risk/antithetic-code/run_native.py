"""Fresh8 paired-noise native observations after closed full64 evidence."""
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
sys.path.insert(0,str(HERE))
from channel import AntitheticChannel
from measure import measure_frame
PREFIX='geometry-risk-antithetic-001'
PROTOCOL=HERE.parent/'antithetic-protocol-001.md'
IDS=['000120','000121','000123','000125','000127','000129','000130','000131']


def sources():
    out=base.sources();out['pilot']=base.source_identity(ROOT,['experiments/geometry-risk/pilot-code'])
    out['antithetic']=base.source_identity(ROOT,['experiments/geometry-risk/antithetic-code'])
    return out


def metadata():
    out=base.metadata();out[str(PROTOCOL.relative_to(ROOT))]=base.sha256(PROTOCOL)
    p=HERE.parent/'antithetic-analysis-clarification-001.md';out[str(p.relative_to(ROOT))]=base.sha256(p)
    return out


def require_pilot():
    manifest=ROOT/'data/runs/geometry-risk-pilot-001.json'
    auditpath=ROOT/'data/provenance/geometry-risk-pilot-001-native-audit.json'
    resultpath=ROOT/'data/analysis/geometry-risk-pilot-001-ridge-001.json'
    run=base.read(manifest);audit=base.read(auditpath);result=base.read(resultpath)
    assert run['state']=='finished64_native_observations_independent_audit_pending'
    assert audit['state']=='passed_full64_independent_native_audit' and audit['native_rows']==8320
    assert audit['manifest_sha256']==base.sha256(manifest)==result['manifest_sha256']
    assert audit['verifier_sha256']==base.sha256(ROOT/'data/provenance/verify-geometry-risk-pilot-001.py')
    assert audit['actual_terminal_PID']==run['pid']==74257 and base.terminal(run['pid'])
    current=sources();current.pop('antithetic')
    assert run['source_identities']==run['source_identities_after']==current
    assert run['readonly_states']==535 and run['optimizer_updates']==0 and run['no_parameter_gradients']
    checks={str(p.relative_to(ROOT)):base.sha256(p) for p in (manifest,auditpath,resultpath)}
    for suffix in ('ridge-audit','ridge-local-audit-001'):
        p=ROOT/'data/provenance'/('geometry-risk-pilot-001-'+suffix+'.json');a=base.read(p)
        assert a['state']=='passed_independent_fixed_ridge_and_frame_bootstrap_audit'
        assert a['result_sha256']==base.sha256(resultpath) and a['native_audit_sha256']==base.sha256(auditpath)
        assert a['verifier_sha256']==base.sha256(ROOT/'data/provenance/verify-geometry-risk-ridge-001.py')
        checks[str(p.relative_to(ROOT))]=base.sha256(p)
    return run,dict(artifacts_sha256=checks,actual_terminal_PID=run['pid'])


def pair_cpu_gate():
    p=ROOT/'data/engineering/geometry-risk-antithetic-sheng-CPU-001.json';r=base.read(p)
    assert r['state']=='passed' and r['tests']==6 and r['CUDA_visible_devices']==''
    for path,h in r['source_sha256'].items():assert base.sha256(ROOT/path)==h
    assert set(r['source_sha256'])=={str(p.relative_to(ROOT)) for p in HERE.glob('*.py')}
    return base.sha256(p)


def main():
    path=ROOT/'data/runs'/f'{PREFIX}.json';directory=Path('/mnt/d/paper6/runs')/PREFIX
    assert not path.exists() and not directory.exists(), 'preserve previous complete or failed observation run'
    lock=base.read(base.INPUT);expected=sources();meta=metadata();cpu=base.cpu_gate()
    previous,eligibility=require_pilot();paired_cpu=pair_cpu_gate()
    assert base.validate_previous(lock)==base.sources()
    selected=IDS
    assert len(selected)==len(set(selected))==8 and not set(selected)&set(previous['fit_ids']+previous['out_of_fit_ids'])
    fold=base.read(ROOT/'data/internal-tuning-fold-001.json')
    train=sorted(fold['folds']['geocomm_tune_train']['ids'])
    assert selected==train[64:72] and not set(selected)&set(fold['folds']['geocomm_tune_holdout']['ids'])
    assert base.sha256(ROOT/'data/internal-tuning-fold-001.json')==lock['fold_sha256']
    assert base.sha256(ROOT/'configs/tuning/stereo_task_seed17_epoch1.yaml')==lock['config_sha256']
    checkpoint=Path('/mnt/d/paper6/runs/stereo-native-task-seed17-002/checkpoint_epoch_1.pth')
    assert base.sha256(checkpoint)==lock['sole_parent_sha256']==previous['sole_parent_sha256']
    used,total=map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(','))
    free=(total-used)*2**20;assert free>=12*2**30, 'physical GPU margin insufficient; no work started'
    directory.mkdir(parents=True)
    report=dict(state='starting',prefix=PREFIX,pid=os.getpid(),started_unix=time.time(),output_directory=str(directory),
                evidence_type='exploratory_training_only_native_risk_observations_no_fit_AP',source_identities=expected,
                metadata_identities=meta,CPU_gate_sha256=cpu,protocol_sha256=base.sha256(PROTOCOL),eligibility=eligibility,
                initial_PCG64_state=np.random.Generator(np.random.PCG64(2804)).bit_generator.state,paired_CPU_gate_sha256=paired_cpu,sole_parent_sha256=lock['sole_parent_sha256'],
                ids=IDS,optimizer_updates=0,
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
        valpath=dataset.root_path/'ImageSets/val.txt'
        assert base.sha256(valpath)==fold['main_validation_split_sha256'] and not set(selected)&set(valpath.read_text().splitlines())
        report['dataset']['mainval_split_sha256']=base.sha256(valpath)
        ids=[str(info['point_cloud']['lidar_idx']) for info in dataset.kitti_infos]
        group=tempfile.TemporaryDirectory(prefix='geometry-risk-antithetic-')
        dist.init_process_group('nccl',init_method='file://'+group.name+'/rank',rank=0,world_size=1)
        model=build_network(config.MODEL,len(config.CLASS_NAMES),dataset).cuda().eval()
        reference=torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state']
        assert len(reference)==len(model.state_dict())==535;model.load_state_dict(reference,strict=True)
        for p in model.parameters():p.requires_grad_(False)
        assert base.assert_frozen(reference,model.state_dict(),set())==535
        observer=base.RiskObserver(model)
        channel=AntitheticChannel(model.backbone_3d.stereo_feature_link,directory/'noise',[(1,2,20,1248),(1,2,20,1248),(1,8,20,156)])
        assert channel.calls==channel.prepared==0 and channel.rng.bit_generator.state==report['initial_PCG64_state']
        report.update(state='running',torch_version=torch.__version__,cuda_runtime=torch.version.cuda,native_layouts=channel.layouts)
        base.save(path,report)
        with (directory/'native-loss.jsonl').open('x') as stream:
            for frame in selected:
                assert sources()==expected and metadata()==meta and base.cpu_gate()==cpu and pair_cpu_gate()==paired_cpu
                result=measure_frame(frame,dataset.collate_batch([dataset[ids.index(frame)]]),model,observer,channel,directory,stream,reference)
                report['frames'][frame]=dict(report_path=str(directory/(frame+'.json')),report_sha256=base.sha256(directory/(frame+'.json')),
                    clean_loss=result['clean_loss'],full_AWGN10_loss=result['full_AWGN10_loss'],
                    source_files={str(dataset.root_path.resolve()/'training'/component/(frame+suffix)):base.sha256(dataset.root_path.resolve()/'training'/component/(frame+suffix))
                                  for component,suffix in [('image_2','.png'),('image_3','.png'),('label_2','.txt'),('calib','.txt')]})
                assert sources()==expected and metadata()==meta and base.cpu_gate()==cpu and pair_cpu_gate()==paired_cpu
                assert torch.cuda.max_memory_reserved()+2*2**30<=free
                report.update(completed_frames=len(report['frames']),completed_native_passes=channel.calls,checked_unix=time.time())
                base.save(path,report)
        calls=dict(steps=8208,student=16416,codec=8208,channel=8208,build_cost=8208,map_to_bev=8208,BEV=8208,head3D=8208,forbidden=0)
        assert observer.calls==calls and channel.calls==channel.prepared==8208 and channel.pending is None
        assert channel.pairs.draws==4104 and channel.pairs.index==0 and channel.pairs.positive is None
        report.update(state='finished8_paired_observations_independent_audit_pending',calls=calls,channel_attempts=8208,
                      attempted_complex_uses=8208*62400,readonly_states=535,all535_final_state_hashes=base.state_hashes(model),
                      no_parameter_gradients=all(p.grad is None for p in model.parameters()),records_sha256=base.sha256(directory/'native-loss.jsonl'),
                      final_PCG64_state=channel.rng.bit_generator.state,independent_noise_draws=channel.pairs.draws,source_identities_after=sources(),
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
