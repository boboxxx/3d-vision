"""Pinned YOLOv5n CPU extraction on complete native stereo sensor IDs."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

import contract as c


def state(model):
    return {key: c.describe(value) for key, value in model.state_dict().items()}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--scope', choices=('engineering', 'main'), required=True)
    args = parser.parse_args(); frame_ids = c.ids(args.scope); source = c.sources()
    prefix = 'public-val-ROI-' + args.scope + '-001'
    directory = c.NATIVE / prefix; manifest = c.ROOT / 'data/runs' / (prefix + '.json')
    assert not directory.exists() and not manifest.exists()
    if args.scope == 'main':
        proof = c.read(c.ROOT / 'data/provenance/public-val-ROI-engineering-001-local-verification.json')
        assert proof['state'] == 'passed_all4_transferred_ROI_view_records' and proof['sources'] == source
    metadata = c.read(c.METADATA); checkpoint = Path(metadata['remote_path'])
    assert c.sha(checkpoint) == metadata['checkpoint_sha256']
    os.environ['YOLOv5_AUTOINSTALL'] = 'false'
    sys.path.insert(0, str(c.ROOT / 'third_party/yolov5'))
    from models.common import DetectMultiBackend
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression, scale_boxes
    import cv2
    torch.set_num_threads(2); torch.manual_seed(17); np.random.seed(17)
    model = DetectMultiBackend(str(checkpoint), device=torch.device('cpu'), fp16=False).eval()
    model.requires_grad_(False)
    assert [model.names[index] for index in (2, 5, 7)] == ['car', 'bus', 'truck']
    initial = state(model)
    allowed = [c.DATA / 'training' / camera / (frame + '.png') for frame in frame_ids for camera in ('image_2', 'image_3')]
    directory.mkdir(); records = directory / 'records.jsonl'
    run = dict(state='running', pid=os.getpid(), scope=args.scope, ordered_ids=frame_ids, sources=source,
               started_unix=time.time(), records_path=str(records), frames_done=0, views_done=0,
               device='cpu', threads=2, initial_states=initial, model_parameters=sum(p.numel() for p in model.parameters()),
               GPU_used=False, GT_calibration_LiDAR_received_images_read=False, source_only_PHY_uses=None,
               source_only_PHY_energy=None, protocol=metadata['ROI_variant'])
    c.save(manifest, run); control = dict(active=True); sys.addaudithook(c.guard(allowed, control))
    try:
        with records.open('x') as stream, torch.inference_mode():
            for index, frame in enumerate(frame_ids, 1):
                views = []
                for camera in ('image_2', 'image_3'):
                    path = c.DATA / 'training' / camera / (frame + '.png'); blob = path.read_bytes()
                    bgr = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
                    assert bgr is not None and bgr.dtype == np.uint8 and bgr.ndim == 3 and bgr.shape[-1] == 3
                    with Image.open(io.BytesIO(blob)) as png:
                        assert png.mode == 'RGB' and np.array_equal(np.asarray(png), bgr[:, :, ::-1])
                    resized = letterbox(bgr, (640, 640), stride=model.stride, auto=True)[0]
                    image = torch.from_numpy(np.ascontiguousarray(resized[:, :, ::-1].transpose(2, 0, 1))).float()[None] / 255
                    assert image.device.type == 'cpu' and not torch.is_grad_enabled() and all(not m.training for m in model.modules())
                    detections = non_max_suppression(model(image), conf_thres=.25, iou_thres=.45,
                                                    classes=[2, 5, 7], agnostic=False, max_det=300)[0]
                    detections[:, :4] = scale_boxes(image.shape[-2:], detections[:, :4], bgr.shape).round()
                    height, width = map(int, bgr.shape[:2]); boxes = []
                    for x1, y1, x2, y2, confidence, category in detections.cpu().tolist():
                        xyxy = [int(value) for value in (x1, y1, x2, y2)]
                        assert 0 <= xyxy[0] <= xyxy[2] <= width and 0 <= xyxy[1] <= xyxy[3] <= height
                        if xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1]:
                            boxes.append(dict(xyxy=xyxy, confidence=confidence, class_id=int(category)))
                    mask = c.mask([height, width], boxes)
                    assert state(model) == initial
                    views.append(dict(camera=camera, image_sha256=hashlib.sha256(blob).hexdigest(),
                                      native_RGB8=c.describe(bgr[:, :, ::-1]), native_hw=[height, width],
                                      letterbox_shape=list(image.shape), actual_YOLO_input=c.describe(image),
                                      boxes=boxes, union_area_pixels=int(mask.sum()), mask_uint8=c.describe(mask)))
                stream.write(json.dumps(dict(frame_id=frame, views=views), allow_nan=False) + '\n'); stream.flush()
                run.update(frames_done=index, views_done=2 * index); c.save(manifest, run)
                if index % 100 == 0:
                    print(json.dumps(dict(scope=args.scope, frames=index)), flush=True)
        assert c.sources() == source and state(model) == initial
        run.update(state='finished', ended_unix=time.time(), records_sha256=c.sha(records), final_states=state(model),
                   readonly_model=True, read_barrier=True, complete_native_PIL_cv2_pixel_identity=True)
        c.save(manifest, run)
    except BaseException as error:
        run.update(state='failed', ended_unix=time.time(), error=repr(error)); c.save(manifest, run); raise
    print(json.dumps(dict(state=run['state'], frames=len(frame_ids), views=2 * len(frame_ids))))


if __name__ == '__main__':
    main()
