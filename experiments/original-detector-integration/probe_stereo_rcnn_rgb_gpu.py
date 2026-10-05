"""Prelocked native Stereo-RCNN received-RGB GPU engineering, no AP."""
import argparse,importlib.util,json,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];STEREO=ROOT/'third_party/Stereo-RCNN'
sys.path[:0]=[str(STEREO/'lib'),str(HERE),str(ROOT/'src'),str(ROOT/'reproduction/cao2025')]
from adapter import stereo_rcnn_preprocess,transmit_rgb
from data import StereoRGB
from wireless import WirelessVariant
import radio
from geocomm.evidence import sha256,source_identity
from geocomm.pooling_diagnostic import state_hashes
from geocomm.stereo_baseline import decode_3d,kitti_line

def check(value,message):
    if not value:raise RuntimeError(message)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output=args.output.resolve()
    if args.output.exists():p.error('preserve prior evidence; unique output required')
    check(json.loads((ROOT/'data/provenance/stereo-native-task-seed17-002-closure.json').read_text())['state']=='closed_all_audits_passed','F6b closure required')
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    check(len(memory)==1,'single GPU required');used,total=map(lambda x:int(x.strip()),memory[0].split(','));free=(total-used)*2**20
    check(free>=12*2**30,'NVIDIA physical free must exceed12GiB; leave original training untouched')
    specs={'project':(ROOT,['src','scripts','configs','pyproject.toml']),
      'liga':(ROOT/'third_party/LIGA-Stereo',['liga','configs','tools','setup.py']),
      'mmdet':(ROOT/'third_party/mmdetection_kitti',['mmdet']),
      'stereo_rcnn':(STEREO,['lib','demo.py','test_net.py'])}
    sources={k:source_identity(*v) for k,v in specs.items()}
    original=json.loads((ROOT/'data/provenance/cao2025-staged-trainer-source-bfee9d8.json').read_text())
    check(len(original)==21 and all(sha256(ROOT/'reproduction/cao2025'/k)==v for k,v in original.items()),'original21 changed')
    original_protocol=sha256(ROOT/'reproduction/cao2025/formal-protocol.md')
    check(original_protocol=='682fb774e09b9be7f68d09f8251dac6d4b99af3290f02660951d398f9e00eace','original protocol changed')
    info=dict(state='starting',scope='one-frame native Stereo-RCNN three RGB endpoints, no AP',started_at_unix=time.time(),
      source_identities=sources,original21_sources=original,original_protocol_sha256=original_protocol,
      NVIDIA_free_before_bytes=free,conditions=[],device='cuda',codec_device='cpu',seed=17)
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(info,indent=2,allow_nan=False)+'\n')
    save();handles=[];dense_module=None;original_align=None
    try:
        torch.set_num_threads(2);torch.manual_seed(17);torch.cuda.manual_seed_all(17);np.random.seed(17)
        from model.stereo_rcnn.resnet import resnet
        from model.utils.config import cfg
        from model.utils.blob import prep_im_for_blob
        from model.utils.kitti_utils import read_obj_calibration
        from model.dense_align import dense_align
        dense_module=dense_align;original_align=dense_align.align_parallel
        binaries=list((STEREO/'lib/model').glob('_C*.so'))
        check(len(binaries)==1 and sha256(binaries[0])=='5c3cbc235cad8e67dbb440cb01940d912f72b702455e51dafde2573f7d9c6429','verified native operator differs')
        model=resnet(np.asarray(['__background__','Car']),101,pretrained=False);model.create_architecture()
        checkpoint=Path('/mnt/d/paper6/checkpoints/stereo-rcnn-branch1-author.pth')
        check(sha256(checkpoint)=='b7a07a897224f75bc30b2d4c06f39927e92933a8bcaad6e679aff605b2346c14','author checkpoint differs')
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)['model'];actual=model.state_dict()
        check(len(actual)==len(state)==670 and set(actual)==set(state),'full670 state coverage')
        check(all(v.shape==actual[k].shape and v.dtype==actual[k].dtype and torch.isfinite(v).all() for k,v in state.items()),'complete shape/dtype/finite')
        model.load_state_dict(state,strict=True)
        check(all(torch.equal(v,state[k]) for k,v in model.state_dict().items()),'strict load values')
        model.cuda().eval();before=state_hashes(model);del state,actual
        check(all(not m.training for m in model.modules()),'native module eval required')
        spec=importlib.util.spec_from_file_location('paper6_cao2025_semantic',ROOT/'reproduction/cao2025/model.py')
        semantic=importlib.util.module_from_spec(spec);spec.loader.exec_module(semantic)
        codec=WirelessVariant(semantic.SemanticVariant('/mnt/d/paper6/checkpoints/spynet_sintel_final-3d2a1287.pth')).cpu().eval()
        initializer=Path('/mnt/d/paper6/runs/cao2025-full-native-seed17-001-stage1/initialization.pth')
        check(sha256(initializer)=='f4d0d5d5bcc203ee5fd0554a0e691b2a530c56cb55d48fe21b6fc0f97331d3cb','fresh initializer differs')
        state=torch.load(initializer,map_location='cpu',weights_only=False)['model_state']
        check(len(state)==len(codec.state_dict())==768,'complete768 codec states')
        codec.load_state_dict(state,strict=True);del state;codec_before=state_hashes(codec)
        def forbidden(*a):raise RuntimeError('privileged training/clean-reconstruction API called')
        codec.forward=forbidden;codec.semantic_mse=forbidden
        handles.append(model.RCNN_proposal_target.register_forward_pre_hook(forbidden))
        cache=StereoRGB('/mnt/d/paper6/data/kitti',ROOT/'data/engineering/cao2025-roi-holdout-001.jsonl',
          ROOT/'data/engineering/cao2025-roi-holdout-audit-001.json',ROOT/'data/internal-tuning-fold-001.json','geocomm_tune_holdout')
        frame=cache[0];check(frame['frame_id']=='000036','firstordered frame')
        calibpath=Path('/mnt/d/paper6/data/kitti/training/calib/000036.txt');calibration=read_obj_calibration(str(calibpath))
        calls={};pending={}
        def entry(module,args):
            check(not module.training and not torch.is_grad_enabled(),'native sensor entry must eval/no_grad')
            check(args[0] is pending['images'][0] and args[1] is pending['images'][1],'received-image bypass at detector entry')
            check(all(torch.count_nonzero(v)==0 for v in args[3:]),'GT placeholders must be zero')
            calls['detector']+=1
        def backbone(module,args):
            check(args[0] is pending['images'][calls['image_backbone']],'received-image bypass at backbone')
            calls['image_backbone']+=1
        def alignment(*a,**kw):
            check(not torch.is_grad_enabled() and a[2] is pending['images'][0] and a[3] is pending['images'][1],'dense alignment clean-image bypass')
            calls['dense_alignment']+=1
            return original_align(*a,**kw)
        handles.extend([model.register_forward_pre_hook(entry),model.RCNN_layer0.register_forward_pre_hook(backbone)])
        dense_align.align_parallel=alignment
        info.update(required_detector_states=670,detector_checkpoint_sha256=sha256(checkpoint),detector_initial_state_hashes=before,
          operator_binary_sha256=sha256(binaries[0]),required_codec_states=768,codec_initialization_sha256=sha256(initializer),
          codec_initial_state_hashes=codec_before,frame_id='000036',calibration_sha256=sha256(calibpath),confidence_threshold=.05)
        gt=torch.zeros(1,5,device='cuda');count=torch.zeros(1,dtype=torch.long,device='cuda');torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for condition in ('clean_relay','identity','awgn'):
                calls.update(detector=0,image_backbone=0,dense_alignment=0);pending.clear()
                result=dict(outputs=(frame['left'],frame['right']),accounting=None,erasure=None,decoder_range=None) if condition=='clean_relay' else transmit_rgb(codec,radio,frame['left'],frame['right'],frame['boxes'],condition,10.,torch.Generator().manual_seed(17))
                row={k:result[k] for k in ('accounting','erasure','decoder_range')};row['condition']=condition
                if result['outputs'] is not None:
                    images,info_cpu,shape=stereo_rcnn_preprocess(result['outputs'],prep_im_for_blob,cfg.PIXEL_MEANS,cfg.TRAIN.SCALES[0],cfg.TRAIN.MAX_SIZE)
                    tensors=tuple(v.cuda() for v in images);native_info=info_cpu.cuda();pending['images']=tensors
                    output=model(*tensors,native_info,gt,gt,gt,gt,gt,count)
                    predictions,counts=decode_3d(output,*tensors,native_info,shape,calibration,threshold=.05);torch.cuda.synchronize()
                    check(calls==dict(detector=1,image_backbone=2,dense_alignment=int(counts['initial_solutions']>0)),'native call counts')
                    lines=[kitti_line(v,calibration) for v in predictions]
                    check(all(len(line.split())==16 and np.isfinite([float(v) for v in line.split()[1:]]).all() for line in lines),'finite complete KITTI prediction rows')
                    row.update(prediction_count=len(predictions),predictions=predictions,native_3D_counts=counts,kitti_lines=lines,
                      processed_image_shape=list(tensors[0].shape),original_image_shape=list(shape),im_info=info_cpu.tolist(),GT_placeholder_only=True)
                    del output,images,tensors,native_info,predictions;pending.clear()
                else:row.update(prediction_count=0,predictions=[],native_3D_counts=dict(detections_2d=0,initial_solutions=0,dense_solutions=0),kitti_lines=[])
                row['calls']=dict(calls)
                check(state_hashes(model)==before and state_hashes(codec)==codec_before,'670/768 states mutated')
                check(torch.cuda.max_memory_reserved()+2*2**30<=free,'physical memory margin failed')
                info['conditions'].append(row);save();print(json.dumps({k:row[k] for k in ('condition','prediction_count','native_3D_counts','calls')}),flush=True)
                del result
        for name,spec in specs.items():check(source_identity(*spec)==sources[name],'source changed: '+name)
        check(all(sha256(ROOT/'reproduction/cao2025'/k)==v for k,v in original.items()),'original sources changed')
        check(sha256(ROOT/'reproduction/cao2025/formal-protocol.md')==original_protocol,'original protocol changed')
        info.update(state='passed',ended_at_unix=time.time(),detector_states_readonly=670,codec_states_readonly=768,
          peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
          prototype_sources=source_identity(HERE,['adapter.py','probe_stereo_rcnn_rgb_gpu.py']),
          protocol_sha256=sha256(HERE/'StereoRCNN-native-GPU-engineering.md'),limitations='One frame, freshuntrained codec, engineering only; no AP/fullfold/latency/trained-baseline claim')
        save()
    except BaseException as exc:info.update(state='failed',exception=repr(exc),ended_at_unix=time.time());save();raise
    finally:
        for handle in handles:handle.remove()
        if dense_module is not None and original_align is not None:dense_module.align_parallel=original_align

if __name__=='__main__':main()
