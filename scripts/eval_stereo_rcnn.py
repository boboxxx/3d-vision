#!/usr/bin/env python3
"""Full fixed KITTI validation with released Stereo-RCNN and complete 3D solve."""
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
STEREO = ROOT/'third_party/Stereo-RCNN'
EVALUATOR = ROOT/'third_party/LIGA-Stereo/liga/datasets/kitti'
sys.path[:0] = [str(STEREO/'lib'),str(ROOT/'src'),str(EVALUATOR)]
from geocomm.evidence import sha256,source_identity,serializable
from geocomm.stereo_baseline import decode_3d,kitti_line


def save(path,value):
    temporary = path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False,default=serializable)+'\n')
    temporary.replace(path)


def preflight(root):
    ids = (root/'ImageSets/val.txt').read_text().split()
    train = (root/'ImageSets/train.txt').read_text().split()
    if len(ids)!=3769 or len(train)!=3712 or len(set(ids))!=3769 or set(ids)&set(train):
        raise RuntimeError('require complete fixed 3712/3769 disjoint split')
    if any(re.fullmatch(r'\d{6}',frame) is None for frame in ids+train):
        raise RuntimeError('invalid KITTI frame id')
    for name,values in [('train',train),('val',ids)]:
        if (STEREO/'data/kitti/splits'/f'{name}.txt').read_text().split()!=values:
            raise RuntimeError('split differs from author detector split')
    manifest = json.loads((root/'download-manifest.json').read_text())
    archives = {}
    for name in ['label_2','calib','image_2','image_3']:
        identity = manifest['archives'].get(f'data_object_{name}.zip')
        if not identity or identity.get('training_files')!=7481 or identity.get('validation')!='ZIP_CRC_and_file_size':
            raise RuntimeError('incomplete or unverified '+name+' archive extraction')
        archives[name] = identity
        extension = '.png' if name.startswith('image') else '.txt'
        if len(list((root/'training'/name).glob('*'+extension)))!=7481:
            raise RuntimeError('full training folder count mismatch: '+name)
        for frame in ids:
            if not (root/'training'/name/(frame+extension)).is_file():
                raise FileNotFoundError(name+'/'+frame)
    return ids,archives


def metrics(root,ids,run):
    # Import the unchanged evaluator without importing unrelated dataset/model modules.
    from kitti_object_eval_python import kitti_common
    from kitti_object_eval_python.eval import get_official_eval_result,do_eval
    frames = [int(frame) for frame in ids]
    gt = kitti_common.get_label_annos(root/'training/label_2',frames)
    dt = kitti_common.get_label_annos(run/'data',frames)
    text,strict = get_official_eval_result(gt,dt,['Car'])
    (run/'evaluator.txt').write_text(text)
    # Original paper says IoU .5, but does not identify recall sampling/version.
    arrays = do_eval(gt,dt,[0],np.full((1,3,1),.5),compute_aos=True)
    relaxed = {}
    for metric,index in [('bbox',0),('bev',1),('3d',2),('aos',3)]:
        for recall,offset in [('R11',0),('R40',4)]:
            if arrays[index+offset] is not None:
                relaxed[f'{metric}_{recall}'] = arrays[index+offset][0,:,0].tolist()
    result = dict(primary_Car_IoU_0_7_R40={key:float(value) for key,value in strict.items()},
                  secondary_Car_IoU_0_5=relaxed,
                  difficulty_order=['easy','moderate','hard'],
                  original_paper_reproduction=False,
                  note='clean released detector; original codec/split/evaluator version still unresolved')
    values = [float(v) for v in strict.values()]
    values += [v for row in relaxed.values() for v in row]
    if not np.isfinite(values).all():
        raise RuntimeError('nonfinite AP metrics')
    save(run/'metrics.json',result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    root,checkpoint,run = args.root.resolve(),args.checkpoint.resolve(),args.run_dir.resolve()
    ids,archives = preflight(root)
    from model.utils.config import cfg
    binaries = list((STEREO/'lib/model').glob('_C*.so'))
    if len(binaries)!=1:
        raise RuntimeError('expected one compiled author operator library')
    identity = dict(dataset_root=str(root),checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
        validation_split_sha256=sha256(root/'ImageSets/val.txt'),
        training_split_sha256=sha256(root/'ImageSets/train.txt'),archives=archives,seed=args.seed,
        complete_detector_cfg=cfg,
        confidence_threshold=.05,
        source=source_identity(ROOT,['src','scripts']),
        detector_source=source_identity(STEREO,['lib']),
        evaluator_source=source_identity(EVALUATOR,['kitti_object_eval_python']),
        operator_binary_sha256=sha256(binaries[0]))
    # Canonical JSON representation makes resumed settings comparable across processes.
    identity = json.loads(json.dumps(identity,default=serializable,allow_nan=False))
    if run.exists() and not args.resume:
        parser.error('retain existing runs; choose a new directory or explicit --resume')
    run.mkdir(parents=True,exist_ok=True)
    with (run/'process.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('this run already has a live evaluator process')
        report_path = run/'run.json'
        if args.resume:
            report = json.loads(report_path.read_text())
            if report['identity']!=identity or report['state']=='finished':
                raise RuntimeError('resume identity mismatch or already finished run')
        else:
            report = dict(state='running',identity=identity,started_unix=time.time(),frames_total=len(ids),
                          dependencies={name:importlib.metadata.version(name) for name in
                              ['torch','numpy','numba','cuda-python','scipy','opencv-python']},
                          gpu=torch.cuda.get_device_name(0),frames_complete=0)
        for folder in ['data','frames']:
            (run/folder).mkdir(exist_ok=True)
        save(report_path,report)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        try:
            from model.stereo_rcnn.resnet import resnet
            from model.utils.blob import prep_im_for_blob
            from model.utils.kitti_utils import read_obj_calibration
            model = resnet(np.asarray(['__background__','Car']),101,pretrained=False)
            model.create_architecture()
            released = torch.load(checkpoint,map_location='cpu',weights_only=False)
            model.load_state_dict(released['model'],strict=True)
            model.cuda().eval()
            count = torch.zeros(1,dtype=torch.long,device='cuda')
            gt = torch.zeros(1,5,device='cuda')  # unused slots in author's evaluation API
            for index,frame in enumerate(ids):
                paths = {name:root/'training'/name/(frame+('.png' if name.startswith('image') else '.txt'))
                         for name in ['image_2','image_3','calib','label_2']}
                inputs_sha256 = {name:sha256(path) for name,path in paths.items()}
                prediction_path,frame_path = run/'data'/f'{frame}.txt',run/'frames'/f'{frame}.json'
                if frame_path.exists():
                    previous = json.loads(frame_path.read_text())
                    if previous['inputs_sha256']!=inputs_sha256 or previous['prediction_sha256']!=sha256(prediction_path):
                        raise RuntimeError('resumed frame identity mismatch: '+frame)
                else:
                    images = [cv2.imread(str(paths[name])) for name in ['image_2','image_3']]
                    if any(image is None for image in images) or images[0].shape!=images[1].shape:
                        raise RuntimeError('invalid stereo pair: '+frame)
                    shape = images[0].shape
                    left,right,scale = prep_im_for_blob(*images,cfg.PIXEL_MEANS,cfg.TRAIN.SCALES[0],cfg.TRAIN.MAX_SIZE)
                    tensors = [torch.from_numpy(im).permute(2,0,1).unsqueeze(0).contiguous().cuda() for im in [left,right]]
                    info = torch.tensor([[*left.shape[:2],scale]],dtype=torch.float32,device='cuda')
                    calibration = read_obj_calibration(str(paths['calib']))
                    start = time.perf_counter()
                    with torch.no_grad():
                        output = model(*tensors,info,gt,gt,gt,gt,gt,count)
                        predictions,counts = decode_3d(output,*tensors,info,shape,calibration)
                    torch.cuda.synchronize()
                    temporary = prediction_path.with_suffix('.txt.part')
                    temporary.write_text(''.join(kitti_line(row,calibration) for row in predictions))
                    temporary.replace(prediction_path)
                    save(frame_path,dict(frame_id=frame,inputs_sha256=inputs_sha256,
                        prediction_sha256=sha256(prediction_path),predictions=predictions,counts=counts,
                        inference_seconds=time.perf_counter()-start,
                        timing_note='inference plus full 3D solve; cold first frame; not formal latency benchmark'))
                report['frames_complete'] = index+1
                if (index+1)%50==0 or index==len(ids)-1:
                    save(report_path,report)
                    print(json.dumps(dict(frames_complete=index+1,total=len(ids))),flush=True)
            if {p.stem for p in (run/'data').glob('*.txt')}!=set(ids):
                raise RuntimeError('prediction frame count/set mismatch')
            report['metrics'] = metrics(root,ids,run)
            report.update(state='finished',ended_unix=time.time())
        except BaseException as error:
            report.update(state='failed',error=repr(error),ended_unix=time.time())
            raise
        finally:
            save(report_path,report)


if __name__=='__main__':
    main()
