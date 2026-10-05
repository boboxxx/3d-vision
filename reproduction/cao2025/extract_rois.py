"""Sensor-only frozen YOLOv5n ROI extraction; no labels/LiDAR/calibration read."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--fold', type=Path, required=True)
    parser.add_argument('--split', choices=['geocomm_tune_train','geocomm_tune_holdout'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--engineering-limit', type=int)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.manifest.json').exists():
        parser.error('preserve earlier records; choose unique output')
    if args.engineering_limit is not None and args.engineering_limit<=0:
        parser.error('positive engineering limit required')
    metadata = json.loads((Path(__file__).parent/'upstream/yolov5.json').read_text())
    assert digest(args.checkpoint)==metadata['checkpoint_sha256'], 'official checkpoint identity differs'
    upstream = ROOT/'third_party/yolov5'
    revision = subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    assert revision==metadata['revision'], 'official YOLO revision differs'
    assert not subprocess.check_output(['git','-C',str(upstream),'diff','--name-only'],text=True).strip()
    os.environ['YOLOv5_AUTOINSTALL']='false'
    sys.path.insert(0,str(upstream))
    # Standalone process avoids the different LIGA/mmdet `utils` namespaces.
    from models.common import DetectMultiBackend
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression, scale_boxes
    import cv2
    torch.set_num_threads(4)
    model = DetectMultiBackend(str(args.checkpoint),device=torch.device('cpu'),fp16=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert [model.names[index] for index in (2,5,7)]==['car','bus','truck']
    fold = json.loads(args.fold.read_text())
    ids = fold['folds'][args.split]['ids']
    selected = ids[:args.engineering_limit] if args.engineering_limit else ids
    started = time.time()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream,torch.inference_mode():
        for frame in selected:
            views = []
            for camera in ('image_2','image_3'):
                path = args.data_root/'training'/camera/(frame+'.png')
                bgr = cv2.imread(str(path))
                if bgr is None:
                    raise FileNotFoundError(path)
                resized = letterbox(bgr,(640,640),stride=model.stride,auto=True)[0]
                rgb_chw = np.ascontiguousarray(resized[...,::-1].transpose(2,0,1))
                image = torch.from_numpy(rgb_chw).float().unsqueeze(0)/255
                detections = non_max_suppression(model(image),conf_thres=.25,iou_thres=.45,
                    classes=[2,5,7],agnostic=False,max_det=300)[0]
                detections[:,:4] = scale_boxes(image.shape[-2:],detections[:,:4],bgr.shape).round()
                h,w = bgr.shape[:2]
                mask = np.zeros((h,w),dtype=np.uint8)
                boxes = []
                for x1,y1,x2,y2,confidence,category in detections.cpu().tolist():
                    x1,y1,x2,y2 = [int(value) for value in (x1,y1,x2,y2)]
                    assert 0<=x1<=x2<=w and 0<=y1<=y2<=h
                    if x2>x1 and y2>y1:
                        boxes.append(dict(xyxy=[x1,y1,x2,y2],confidence=confidence,class_id=int(category)))
                        mask[y1:y2,x1:x2]=1
                views.append(dict(camera=camera,image_sha256=digest(path),shape=[h,w],
                    letterbox_shape=list(image.shape),boxes=boxes,union_area_pixels=int(mask.sum()),
                    mask_uint8_sha256=hashlib.sha256(mask.tobytes()).hexdigest()))
            stream.write(json.dumps(dict(frame_id=frame,views=views),allow_nan=False)+'\n')
            stream.flush()
    record=dict(state='finished',evidence_type='sensor_only_YOLO_ROI_engineering' if args.engineering_limit else 'sensor_only_YOLO_ROI_fold_extraction',
        split=args.split,frames=len(selected),fold_sha256=digest(args.fold),
        checkpoint_sha256=digest(args.checkpoint),upstream_revision=revision,
        source_sha256=digest(__file__),metadata_sha256=digest(Path(__file__).parent/'upstream/yolov5.json'),
        records_sha256=digest(args.output),device='cpu',torch_version=torch.__version__,
        pretrained_model_parameters=sum(parameter.numel() for parameter in model.parameters()),
        inputs=['left_RGB','right_RGB'],labels_LiDAR_calibration_read=False,
        protocol=metadata['ROI_variant'],elapsed_seconds=time.time()-started,
        limitations='documented ROI variant; not Cao exact thresholds/weights; no channel/control cost or detector AP established')
    args.output.with_suffix('.manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':
    main()
