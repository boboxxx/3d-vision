#!/usr/bin/env python3
"""Independent numerical checks of the actual upstream GPU operators.

These are engineering evidence, never KITTI accuracy or reproduction results.
The reference cost volume uses explicit tensor indexing and linear interpolation.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "third_party/LIGA-Stereo"),
               str(ROOT / "third_party/mmdetection_kitti")]


def reference_volume(left, right, shift, stride):
    n, c, h, w = left.shape
    rows = torch.arange(0, h, stride, device=left.device)
    cols = torch.arange(0, w, stride, device=left.device, dtype=left.dtype)
    sampled_left = left[:, :, ::stride, ::stride]
    sampled_right = right[:, :, rows, :]
    levels = []
    for d in range(shift.shape[1]):
        x = cols[None, :] - shift[:, d, None]
        mask = (x >= 0) & (x <= w - 1)
        lo = x.floor().long().clamp(0, w - 1)
        hi = (lo + 1).clamp(0, w - 1)
        index_shape = (n, c, len(rows), len(cols))
        low = sampled_right.gather(3, lo[:, None, None, :].expand(index_shape))
        high = sampled_right.gather(3, hi[:, None, None, :].expand(index_shape))
        frac = (x - x.floor())[:, None, None, :]
        warped = (low * (1 - frac) + high * frac) * mask[:, None, None, :]
        levels.append(torch.cat([sampled_left, warped], dim=1))
    return torch.stack(levels, dim=2)


def cost_volume():
    from liga.ops.build_cost_volume import build_cost_volume
    results = []
    for dtype in [torch.float32, torch.float64]:
        for stride in [1, 2, 4]:
            torch.manual_seed(17)
            left = torch.randn(2, 3, 8, 12, device="cuda", dtype=dtype, requires_grad=True)
            right = torch.randn_like(left, requires_grad=True)
            shift = torch.tensor([[0, .25, 1, 3.5, 11, 15],
                                  [0, .75, 2, 5.5, 11, 16]], device="cuda", dtype=dtype)
            ref_left = left.detach().clone().requires_grad_()
            ref_right = right.detach().clone().requires_grad_()
            actual = build_cost_volume(left, right, shift, stride)
            expected = reference_volume(ref_left, ref_right, shift, stride)
            upstream_grad = torch.randn_like(actual)
            actual.backward(upstream_grad)
            expected.backward(upstream_grad)
            atol = 3e-5 if dtype == torch.float32 else 1e-11
            errors = {}
            for label, a, b in [("forward", actual, expected),
                                ("grad_left", left.grad, ref_left.grad),
                                ("grad_right", right.grad, ref_right.grad)]:
                torch.testing.assert_close(a, b, atol=atol, rtol=atol)
                errors[label] = float((a - b).abs().max())
            results.append(dict(dtype=str(dtype), stride=stride, max_absolute_errors=errors))
    return results


def voxelizer():
    from geocomm.compat import VoxelGenerator
    v = VoxelGenerator([1, 1, 1], [0, 0, 0, 4, 4, 4], 2, 32)
    points = np.array([[.1, .2, .3], [.4, .5, .6], [1.1, 2.2, 3.3],
                       [4, 0, 0], [-.1, 0, 0]], dtype=np.float32)
    voxels, coords, counts = v.generate(points)
    np.testing.assert_array_equal(coords, [[0, 0, 0], [3, 2, 1]])
    np.testing.assert_array_equal(counts, [2, 1])
    np.testing.assert_allclose(voxels[0], points[:2])
    saved = voxels.copy()
    v.generate(np.array([[2.1, 1.1, .1]], dtype=np.float32))
    np.testing.assert_array_equal(voxels, saved)
    return dict(voxels=len(coords), coordinate_order="ZYX", owns_storage=True)


def sparse_convolution():
    import spconv.pytorch as spconv
    from geocomm.compat import adapt_spconv_state
    torch.manual_seed(17)
    conv = spconv.SubMConv3d(2, 5, 3, padding=1, bias=False).cuda()
    legacy = torch.randn(3, 3, 3, 2, 5, device="cuda")
    converted = adapt_spconv_state(conv, {"weight": legacy})["weight"]
    conv.load_state_dict({"weight": converted}, strict=True)
    shape = (4, 5, 6)
    zz, yy, xx = torch.meshgrid(*[torch.arange(s, device="cuda") for s in shape], indexing="ij")
    indices = torch.stack([torch.zeros_like(zz), zz, yy, xx], -1).reshape(-1, 4).int()
    features = torch.randn(indices.shape[0], 2, device="cuda", requires_grad=True)
    ref_features = features.detach().clone().requires_grad_()
    ref_weight = converted.detach().clone().permute(0, 4, 1, 2, 3).contiguous().requires_grad_()
    sparse = spconv.SparseConvTensor(features, indices, shape, 1)
    actual = conv(sparse).dense()
    dense = ref_features.reshape(*shape, 2).permute(3, 0, 1, 2)[None]
    expected = F.conv3d(dense, ref_weight, padding=1)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    errors = {}
    for label, a, b in [("forward", actual, expected),
                        ("grad_features", features.grad, ref_features.grad),
                        ("grad_kernel", conv.weight.grad, ref_weight.grad.permute(0, 2, 3, 4, 1))]:
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
        errors[label] = float((a - b).abs().max())
    return dict(legacy_shape=list(legacy.shape), runtime_shape=list(conv.weight.shape),
                max_absolute_errors=errors)


def boxes():
    from liga.ops.iou3d_nms import iou3d_nms_utils as iou
    from liga.ops.roiaware_pool3d import roiaware_pool3d_utils as roi
    box = torch.tensor([[0, 0, 0, 2, 2, 2, 0], [1, 0, 0, 2, 2, 2, 0],
                        [5, 0, 0, 2, 2, 2, 0]], dtype=torch.float32, device="cuda")
    expected = torch.tensor([[1, 1/3, 0], [1/3, 1, 0], [0, 0, 1]], device="cuda")
    actual = iou.boxes_iou3d_gpu(box, box)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    keep, _ = iou.nms_gpu(box, torch.tensor([.9, .8, .7], device="cuda"), .25)
    torch.testing.assert_close(keep, torch.tensor([0, 2], device="cuda"))
    points = torch.tensor([[[0, 0, 0], [5, 0, 0], [9, 0, 0]]], dtype=torch.float32, device="cuda")
    ids = roi.points_in_boxes_gpu(points, box[[0, 2]][None])
    torch.testing.assert_close(ids, torch.tensor([[0, 1, -1]], dtype=torch.int32, device="cuda"))
    return dict(iou_max_absolute_error=float((actual - expected).abs().max()),
                nms_kept=keep.tolist(), point_box_ids=ids.tolist())


def evaluator_fixture():
    from liga.datasets.kitti.kitti_object_eval_python.eval import get_official_eval_result
    annotation = dict(name=np.array(["Car"]), truncated=np.array([0.]), occluded=np.array([0]),
        alpha=np.array([0.]), bbox=np.array([[550., 100., 700., 200.]]),
        dimensions=np.array([[3.9, 1.6, 1.6]]), location=np.array([[0., 0., 20.]]),
        rotation_y=np.array([0.]))
    detections = dict(annotation, score=np.array([.9]))
    # Enough positives to populate the 40 recall points; all predictions exact.
    _, metrics = get_official_eval_result([annotation for _ in range(100)],
                                         [detections for _ in range(100)], ["Car"])
    checked = {key: float(value) for key, value in metrics.items() if "3d" in key.lower() and "R40" in key}
    if len(checked) != 3 or any(abs(value - 100.) > 1e-6 for value in checked.values()):
        raise RuntimeError("perfect-detection synthetic evaluator fixture failed: " + repr(checked))
    return dict(fixture="100 synthetic exact Car predictions, not KITTI experiments",
                expected_AP_R40_percent=100, checked_metrics=checked)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("output already exists; retain each verification attempt")
    info = dict(evidence_type="engineering_only", torch=torch.__version__,
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                started_at_unix=time.time(), checks={}, state="running")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def snapshot():
        args.output.write_text(json.dumps(info, indent=2), encoding="utf-8")
    snapshot()
    try:
        for name, check in [("cost_volume", cost_volume), ("voxelizer", voxelizer),
                            ("sparse_convolution", sparse_convolution), ("boxes", boxes),
                            ("evaluator_fixture", evaluator_fixture)]:
            info["current_check"] = name
            snapshot()
            info["checks"][name] = check()
            snapshot()
            print(name + " passed", flush=True)
        info["state"] = "passed"
    except BaseException as exc:
        info["state"] = "failed"
        info["exception"] = repr(exc)
        raise
    finally:
        info["ended_at_unix"] = time.time()
        snapshot()


if __name__ == "__main__":
    main()
