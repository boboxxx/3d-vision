#!/usr/bin/env python3
"""Independent operator checks and full author detector inference; never AP."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'third_party/Stereo-RCNN'
sys.path[:0] = [str(UPSTREAM / 'lib'), str(ROOT / 'src')]
from geocomm.evidence import sha256, source_identity, serializable


def roi_reference(image, rois, size, scale, ratio):
    """Differentiable scalar quadrature, independent of compiled operators."""
    h, w = image.shape[-2:]
    outputs = []
    for roi in rois.tolist():
        batch, x0, y0, x1, y1 = roi
        x0, y0, x1, y1 = [v * scale for v in (x0, y0, x1, y1)]
        rw, rh = max(x1-x0, 1.), max(y1-y0, 1.)
        gh = ratio or math.ceil(rh/size[0])
        gw = ratio or math.ceil(rw/size[1])
        rows = []
        for py in range(size[0]):
            columns = []
            for px in range(size[1]):
                value = image[int(batch), :, 0, 0] * 0
                for iy in range(gh):
                    for ix in range(gw):
                        y = y0 + (py+(iy+.5)/gh)*rh/size[0]
                        x = x0 + (px+(ix+.5)/gw)*rw/size[1]
                        if y < -1 or y > h or x < -1 or x > w:
                            continue
                        y, x = min(max(y, 0), h-1), min(max(x, 0), w-1)
                        yl, xl = int(y), int(x)
                        yh, xh = min(yl+1,h-1), min(xl+1,w-1)
                        dy, dx = y-yl, x-xl
                        value = value + ((1-dy)*(1-dx)*image[int(batch),:,yl,xl]
                            + (1-dy)*dx*image[int(batch),:,yl,xh]
                            + dy*(1-dx)*image[int(batch),:,yh,xl]
                            + dy*dx*image[int(batch),:,yh,xh])/(gh*gw)
                columns.append(value)
            rows.append(torch.stack(columns, -1))
        outputs.append(torch.stack(rows, -2))
    return torch.stack(outputs)


def nms_reference(boxes, scores, threshold):
    order = np.argsort(-scores)
    keep = []
    while len(order):
        index, rest = int(order[0]), order[1:]
        keep.append(index)
        a, b = boxes[index], boxes[rest]
        intersection = np.maximum(np.minimum(a[2:], b[:,2:]) - np.maximum(a[:2],b[:,:2])+1,0).prod(1)
        area = (a[2:]-a[:2]+1).prod()
        other_area = (b[:,2:]-b[:,:2]+1).prod(1)
        order = rest[intersection/(area+other_area-intersection) <= threshold]
    return sorted(keep)  # Original GPU NMS returns original indices ascending.


def operator_checks():
    from model.roi_layers import nms
    from model.roi_layers.roi_align import roi_align
    rows = []
    rois = torch.tensor([[0, .3, .7, 7.4, 5.8], [1,-2,-1.2,2.8,4.4],
                         [0,6.8,5.6,9.,8.], [1,3.,2.,2.,1.]], device='cuda')
    for dtype in (torch.float32, torch.float64):
        for ratio in (0,2):
            image = torch.randn(2,3,7,9,dtype=dtype,device='cuda',requires_grad=True)
            roi = rois.to(dtype)
            expected_input = image.detach().cpu().requires_grad_()
            expected = roi_reference(expected_input, roi.cpu(), (3,4), 1., ratio)
            actual = roi_align(image, roi, (3,4), 1., ratio)
            weights = torch.randn_like(actual)
            (expected*weights.cpu()).sum().backward()
            (actual*weights).sum().backward()
            tolerance = 2e-5 if dtype == torch.float32 else 1e-10
            torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=tolerance)
            torch.testing.assert_close(image.grad.cpu(), expected_input.grad, rtol=tolerance, atol=tolerance)
            rows.append(dict(dtype=str(dtype), sampling_ratio=ratio,
                             forward_max_error=float((actual.cpu()-expected).abs().max()),
                             gradient_max_error=float((image.grad.cpu()-expected_input.grad).abs().max())))
    # Cross the 64-box bitmask boundary and use an inclusive-area-sensitive pair.
    boxes = torch.rand(137,4,device='cuda')*20
    boxes[:,2:] = boxes[:,:2]+torch.rand(137,2,device='cuda')*5+.1
    boxes[:2] = torch.tensor([[0,0,1,1],[1,0,2,1]],device='cuda')
    scores = torch.linspace(1,.01,137,device='cuda')
    expected = nms_reference(boxes.cpu().numpy(),scores.cpu().numpy(),.3)
    actual = nms(boxes,scores,.3).cpu().tolist()
    assert actual == expected, (actual, expected)
    assert nms(boxes[:0], scores[:0], .3).numel() == 0
    return dict(roi_align=rows, inclusive_pixel_nms=dict(boxes=137, kept=len(actual), empty_passed=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('preserve verification evidence; use a unique output path')
    report = dict(evidence_type='engineering_only', state='running', started_at_unix=time.time(),
                  upstream_revision='4d6dd65049f52c1a5f2b6ad716a7e0da5cb02cb3',
                  source=source_identity(UPSTREAM,['lib']), checkpoint_sha256=sha256(args.checkpoint))
    torch.manual_seed(17)
    np.random.seed(17)
    try:
        report['operators'] = operator_checks()
        from model.stereo_rcnn.resnet import resnet
        from model.utils.config import cfg
        model = resnet(np.asarray(['__background__','Car']),101,pretrained=False)
        model.create_architecture()
        checkpoint = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        state = checkpoint['model']
        expected = model.state_dict()
        missing = [k for k in expected if k not in state and not k.endswith('num_batches_tracked')]
        unexpected = [k for k in state if k not in expected]
        mismatched = [k for k in state if k in expected and state[k].shape != expected[k].shape]
        report['checkpoint'] = dict(tensors=len(state), expected_tensors=len(expected), missing=missing,
            unexpected=unexpected, mismatched=mismatched,
            legacy_bn_counters_initialized=[k for k in expected if k not in state and k.endswith('num_batches_tracked')],
            epoch=checkpoint.get('epoch'))
        if missing or unexpected or mismatched:
            raise RuntimeError('incomplete author checkpoint')
        model.load_state_dict(state,strict=True)
        model.cuda().eval()
        images = [cv2.imread(str(UPSTREAM/'demo'/name)).astype(np.float32) for name in ('left.png','right.png')]
        original_shape = images[0].shape
        scale = float(cfg.TRAIN.SCALES[0])/min(images[0].shape[:2])
        # Match the author's in-place subtraction: PIXEL_MEANS is float64.
        for image in images:
            image -= cfg.PIXEL_MEANS
        images = [cv2.resize(im,None,fx=scale,fy=scale,
                            interpolation=cv2.INTER_LINEAR) for im in images]
        inputs = [torch.from_numpy(im).permute(2,0,1).unsqueeze(0).contiguous().cuda() for im in images]
        info = torch.tensor([[*images[0].shape[:2],scale]],dtype=torch.float32,device='cuda')
        gt = torch.zeros(1,device='cuda')
        count = torch.zeros(1,dtype=torch.long,device='cuda')
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.no_grad():
            output = model(*inputs,info,gt,gt,gt,gt,gt,count)
        torch.cuda.synchronize()
        for tensor in output[:8]:
            assert torch.isfinite(tensor).all()
        from geocomm.stereo_baseline import decode_3d
        from model.utils.kitti_utils import read_obj_calibration
        with torch.no_grad():
            predictions, counts = decode_3d(output,*inputs,info,original_shape,
                read_obj_calibration(str(UPSTREAM/'demo/calib.txt')))
        if counts['dense_solutions'] == 0:
            raise RuntimeError('author demo did not exercise successful dense 3D alignment')
        report['demo_3d'] = dict(counts=counts,predictions=predictions,
            calibration_sha256=sha256(UPSTREAM/'demo/calib.txt'),
            image_sha256={name:sha256(UPSTREAM/'demo'/name) for name in ('left.png','right.png')},
            downstream_source=source_identity(ROOT,['src/geocomm/stereo_baseline.py']))
        report['full_detector'] = dict(parameters=sum(p.numel() for p in model.parameters()),
            input_shape=list(inputs[0].shape), output_shapes=[list(t.shape) for t in output[:8]],
            elapsed_seconds=time.perf_counter()-start, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            timing_note='single cold demo inference, not a latency benchmark',
            input_note='author demo stereo RGB; dummy unused GT slots at evaluation',
            dense_3D_alignment='complete author solve, dense disparity enumeration and rectification passed')
        report['state'] = 'passed'
    except Exception as error:
        report.update(state='failed',error=repr(error))
        raise
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,default=serializable,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
