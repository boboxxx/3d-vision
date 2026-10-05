#!/usr/bin/env python3
"""Measure the full executed detector and communication split on fixed tensors.

Convolution/linear MACs are exact for executed dense layers. They exclude custom
cost-volume interpolation, normalization, pooling, activations, KKT and channel
arithmetic: reported FLOPs are explicitly a lower bound, not total FLOPs.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
LIGA = ROOT/'third_party/LIGA-Stereo'
sys.path[:0] = [str(ROOT/'src'),str(LIGA),str(ROOT/'third_party/mmdetection_kitti')]
from geocomm.compat import adapt_spconv_state
from geocomm.evidence import sha256,source_identity,serializable


def profile(config_path,checkpoint_path,calibration_path,warmup,repeats):
    from easydict import EasyDict
    from liga.config import cfg_from_yaml_file
    from liga.models import build_network
    from liga.utils.calibration_kitti import Calibration
    config = cfg_from_yaml_file(str(config_path),EasyDict())
    data = config.DATA_CONFIG
    pc,voxel,stereo_voxel = [np.asarray(v,dtype=np.float32) for v in
        [data.POINT_CLOUD_RANGE,data.VOXEL_SIZE,data.STEREO_VOXEL_SIZE]]
    dataset = SimpleNamespace(class_names=config.CLASS_NAMES,point_cloud_range=pc,voxel_size=voxel,
        grid_size=np.round((pc[3:]-pc[:3])/voxel).astype(int),stereo_voxel_size=stereo_voxel,
        stereo_grid_size=np.round((pc[3:]-pc[:3])/stereo_voxel).astype(int),
        boxes_gt_in_cam2_view=data.BOXES_GT_IN_CAM2_VIEW,point_feature_encoder=SimpleNamespace(num_point_features=3))
    torch.manual_seed(17)
    np.random.seed(17)
    model = build_network(config.MODEL,len(config.CLASS_NAMES),dataset).cuda().eval()
    checkpoint = torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    state = adapt_spconv_state(model,checkpoint['model_state'])
    expected = model.state_dict()
    missing = [name for name in expected if 'semantic_link' not in name and name not in state]
    if missing or any(name not in expected or expected[name].shape!=tensor.shape for name,tensor in state.items()):
        raise RuntimeError('incomplete detector checkpoint')
    expected.update(state)
    model.load_state_dict(expected,strict=True)
    del checkpoint,state,expected
    link = getattr(model,'rgb_semantic_link',None) or model.backbone_3d.semantic_link
    left,right = [torch.randn(1,3,320,1280,device='cuda') for _ in range(2)]
    calibration = Calibration(str(calibration_path))
    calibration.offset(0,55)
    measuring,phase = False,'transmitter' if link is not None else 'full_detector'
    calls = []
    tx_end = torch.cuda.Event(enable_timing=True)
    def boundary(module,inputs):
        nonlocal phase
        tx_end.record()
        phase = 'receiver'
    handles = [link.channel.register_forward_pre_hook(boundary)] if link is not None else []
    def dense_hook(name):
        def count(module,inputs,output):
            if not measuring:
                return
            if isinstance(module,torch.nn.Linear):
                macs = output.numel()*module.in_features
            else:
                macs = output.numel()*(module.in_channels//module.groups)*int(np.prod(module.kernel_size))
            calls.append(dict(module=name,kind=type(module).__name__,phase=phase,
                input_shape=list(inputs[0].shape),output_shape=list(output.shape),MACs=macs))
        return count
    for name,module in model.named_modules():
        if isinstance(module,(torch.nn.Conv2d,torch.nn.Conv3d,torch.nn.Linear)):
            handles.append(module.register_forward_hook(dense_hook(name)))
    samples = []
    teacher_calls = []
    handles.append(model.lidar_model.register_forward_hook(lambda *args:teacher_calls.append(1)))
    feature_teacher_calls = []
    student_enabled = getattr(model.backbone_3d,'student_semantic_link_encoder',None) is not None
    if student_enabled:
        for module in [model.backbone_3d.feature_backbone,model.backbone_3d.feature_neck]:
            handles.append(module.register_forward_hook(lambda *args:feature_teacher_calls.append(1)))
    torch.cuda.reset_peak_memory_stats()
    for iteration in range(warmup+repeats):
        phase = 'transmitter' if link is not None else 'full_detector'
        measuring = iteration==warmup
        batch = dict(batch_size=1,left_img=left,right_img=right,calib=[calibration],image_shape=np.array([[320,1280]]))
        start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        start.record()
        with torch.no_grad():
            predictions,diagnostics = model(batch)
        end.record()
        torch.cuda.synchronize()
        if iteration>=warmup:
            samples.append(dict(wall_ms=(time.perf_counter()-wall_start)*1000,
                stream_elapsed_ms=start.elapsed_time(end),
                transmitter_stream_ms=start.elapsed_time(tx_end) if link is not None else None,
                receiver_including_channel_stream_ms=tx_end.elapsed_time(end) if link is not None else None,
                predictions=int(predictions[0]['pred_boxes'].shape[0])))
        del predictions,diagnostics,batch
    if teacher_calls:
        raise RuntimeError('LiDAR teacher executed during sensor-only inference')
    if feature_teacher_calls:
        raise RuntimeError('original feature teacher executed during student inference')
    for handle in handles:
        handle.remove()
    by_phase = {name:sum(call['MACs'] for call in calls if call['phase']==name)
                for name in {call['phase'] for call in calls}}
    result = dict(config_sha256=sha256(config_path),config=str(config_path),
        parameters=sum(p.numel() for p in model.parameters()),
        untrained_random_codec=link is not None,teacher_forward_calls=len(teacher_calls),
        student_enabled=student_enabled,feature_teacher_forward_calls=len(feature_teacher_calls),
        dense_MACs_by_phase=by_phase,dense_FLOPs_lower_bound_by_phase={k:2*v for k,v in by_phase.items()},
        dense_layer_calls=calls,warmup=warmup,repeats=repeats,samples=samples,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        accounting=link.last_accounting if link is not None else None,
        median_wall_ms=float(np.median([v['wall_ms'] for v in samples])),
        median_transmitter_stream_ms=float(np.median([v['transmitter_stream_ms'] for v in samples])) if link is not None else None)
    del link,model,left,right
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs',nargs='+',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--warmup',type=int,default=3)
    parser.add_argument('--repeats',type=int,default=20)
    args = parser.parse_args()
    if args.output.exists() or args.warmup<1 or args.repeats<2:
        parser.error('unique output, positive warmup and at least two repeats required')
    configs = [p.resolve() for p in args.configs]
    checkpoint,calibration,output = args.checkpoint.resolve(),args.calibration.resolve(),args.output.resolve()
    report = dict(evidence_type='engineering_compute_profile_only',state='running',
        checkpoint_sha256=sha256(checkpoint),calibration_sha256=sha256(calibration),
        project_source=source_identity(ROOT,['src','scripts','configs']),
        detector_source=source_identity(LIGA,['liga']),gpu=torch.cuda.get_device_name(0),
        torch=torch.__version__,threads=torch.get_num_threads(),input_shape=[1,3,320,1280],profiles=[],
        limitations='fixed synthetic RGB; random new codecs; outputs can affect NMS timing; no AP or data loader/I/O; dense FLOPs are lower bounds; host launch gaps included in stream elapsed times')
    os.chdir(LIGA)
    try:
        with tempfile.TemporaryDirectory() as directory:
            dist.init_process_group('gloo',init_method='file://'+directory+'/group',rank=0,world_size=1)
            for config in configs:
                report['profiles'].append(profile(config,checkpoint,calibration,args.warmup,args.repeats))
            dist.destroy_process_group()
        report['state'] = 'passed'
    except BaseException as error:
        report.update(state='failed',error=repr(error))
        if dist.is_initialized():
            dist.destroy_process_group()
        raise
    finally:
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2,default=serializable,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
