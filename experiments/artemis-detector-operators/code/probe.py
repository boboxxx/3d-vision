"""Actual sm120 execution versus independent interpolation and geometric fixtures."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch
from build import ROOT, HERE, INPUT, sha, sources, run


def load(name, binaries):
    selected = [ROOT / p for p in binaries if Path(p).name.startswith(name + '.')]
    assert len(selected) == 1
    spec = importlib.util.spec_from_file_location(name, selected[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(left, right, shifts, downsample):
    # Independent CPU gather and interpolation; no native wrapper or CUDA.
    h, w = left.shape[-2:]
    columns = torch.arange(0, w, downsample, dtype=left.dtype)
    volumes = []
    for shift in shifts[0]:
        pos = columns - shift
        valid = (pos >= 0) & (pos <= w - 1)
        bounded = pos.clamp(0, w - 1)
        lo = bounded.floor().long()
        hi = (lo + 1).clamp(max=w - 1)
        frac = bounded - lo
        shifted = (right[:, :, ::downsample, lo] * (1 - frac) +
                   right[:, :, ::downsample, hi] * frac) * valid
        volumes.append(torch.cat([left[:, :, ::downsample, ::downsample], shifted], dim=1))
    return torch.stack(volumes, dim=2)


def cost_check(module, directory, downsample):
    left = (torch.arange(96, dtype=torch.float32).reshape(1, 2, 6, 8) / 29).requires_grad_()
    right = torch.cos(torch.arange(96, dtype=torch.float32).reshape(1, 2, 6, 8) / 7).requires_grad_()
    shifts = torch.tensor([[0., 1., 1.5]], dtype=torch.float32)
    expected = reference(left, right, shifts, downsample)
    gradient = torch.sin(torch.arange(expected.numel(), dtype=torch.float32).reshape(expected.shape) / 5).contiguous()
    (expected * gradient).sum().backward()
    observed = module.build_cost_volume_forward(left.detach().cuda(), right.detach().cuda(), shifts.cuda(), downsample)
    gl, gr = module.build_cost_volume_backward(gradient.cuda(), shifts.cuda(), downsample)
    torch.cuda.synchronize()
    errors = {}
    for name, actual, target in [('forward', observed, expected), ('left_gradient', gl, left.grad), ('right_gradient', gr, right.grad)]:
        actual = actual.cpu()
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, target.detach(), atol=2e-5, rtol=2e-5)
        errors[name] = float((actual - target.detach()).abs().max())
    path = directory / f'cost-downsample-{downsample}.npz'
    np.savez(path, left=left.detach().numpy(), right=right.detach().numpy(), shifts=shifts.numpy(),
             incoming_gradient=gradient.numpy(), expected_forward=expected.detach().numpy(), actual_forward=observed.cpu().numpy(),
             expected_left_gradient=left.grad.numpy(), actual_left_gradient=gl.cpu().numpy(),
             expected_right_gradient=right.grad.numpy(), actual_right_gradient=gr.cpu().numpy())
    return {'downsample': downsample, 'max_absolute_errors': errors, 'fixture_sha256': sha(path)}


def geometry_checks(iou, roi, directory):
    boxes = torch.tensor([[0,0,0,2,2,2,0], [0,0,0,2,2,2,0], [1,0,0,2,2,2,0], [5,0,0,2,2,2,0]], dtype=torch.float32)
    expected_overlap = torch.tensor([[4,4,2,0], [4,4,2,0], [2,2,4,0], [0,0,0,4]], dtype=torch.float32)
    expected_iou = expected_overlap / (8 - expected_overlap)
    observed_overlap = torch.zeros((4,4), device='cuda')
    observed_iou = torch.zeros_like(observed_overlap)
    iou.boxes_overlap_bev_gpu(boxes.cuda(), boxes.cuda(), observed_overlap)
    iou.boxes_iou_bev_gpu(boxes.cuda(), boxes.cuda(), observed_iou)
    cpu_iou = torch.zeros((4,4))
    iou.boxes_iou_bev_cpu(boxes, boxes, cpu_iou)
    keep = torch.empty(4, dtype=torch.int64)
    count = iou.nms_gpu(boxes.cuda(), keep, 0.5)
    torch.cuda.synchronize()
    torch.testing.assert_close(observed_overlap.cpu(), expected_overlap, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(observed_iou.cpu(), expected_iou, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(cpu_iou, expected_iou, atol=2e-5, rtol=2e-5)
    assert keep[:count].tolist() == [0,2,3]
    roi_boxes = boxes[[0,3]].contiguous()
    points = torch.tensor([[0,0,0], [.9,.9,.9], [3,0,0], [5,0,.5], [-1.2,0,0]], dtype=torch.float32)
    # Bounds are deliberately away from surfaces, so boundary conventions do not decide the test.
    membership = ((points[None] - roi_boxes[:,None,:3]).abs() < roi_boxes[:,None,3:6] / 2).all(dim=-1).int()
    expected_ids = torch.tensor([[0,0,-1,1,-1]], dtype=torch.int32)
    cpu_membership = torch.zeros_like(membership)
    gpu_ids = torch.full((1,len(points)), -1, dtype=torch.int32, device='cuda')
    roi.points_in_boxes_cpu(roi_boxes, points, cpu_membership)
    roi.points_in_boxes_gpu(roi_boxes[None].cuda(), points[None].cuda(), gpu_ids)
    torch.cuda.synchronize()
    assert torch.equal(cpu_membership, membership) and torch.equal(gpu_ids.cpu(), expected_ids)
    path = directory / 'geometric-fixtures.npz'
    np.savez(path, boxes=boxes.numpy(), expected_overlap=expected_overlap.numpy(), actual_overlap=observed_overlap.cpu().numpy(),
             expected_iou=expected_iou.numpy(), actual_iou=observed_iou.cpu().numpy(), cpu_iou=cpu_iou.numpy(),
             nms_keep=keep[:count].numpy(), roi_boxes=roi_boxes.numpy(), points=points.numpy(),
             expected_membership=membership.numpy(), actual_cpu_membership=cpu_membership.numpy(),
             expected_gpu_ids=expected_ids.numpy(), actual_gpu_ids=gpu_ids.cpu().numpy())
    return {'NMS_keep': keep[:count].tolist(), 'fixture_sha256': sha(path), 'ROI_membership': 'passed', 'BEV_overlap_and_IoU': 'passed'}


def main():
    output = ROOT / 'data/engineering/artemis-detector-operators-GPU-001.json'
    assert not output.exists(), 'preserve prior attempt'
    lock = json.loads(INPUT.read_text())
    report = {'state': 'running', 'job_id': os.environ.get('SLURM_JOB_ID'), 'started_unix': time.time(),
              'input_sha256': sha(INPUT), 'execution_sources': {str(p.relative_to(ROOT)): sha(p) for p in HERE.glob('*.py')}}
    try:
        assert torch.cuda.is_available()
        report['GPU'] = torch.cuda.get_device_name()
        report['capability'] = list(torch.cuda.get_device_capability())
        assert 'RTX PRO 6000' in report['GPU'] and report['capability'] == [12,0]
        report['nvidia_smi'] = run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'])
        report['torch_version'] = torch.__version__
        report['torch_cuda'] = torch.version.cuda
        assert torch.__version__ == '2.7.1+cu128'
        build = json.loads((ROOT / 'data/engineering/artemis-detector-operators-build-001.json').read_text())
        assert build['state'] == 'passed_build_only_GPU_unverified'
        report['build_manifest_sha256'] = sha(ROOT / 'data/engineering/artemis-detector-operators-build-001.json')
        report['sources_before'] = sources(lock)
        report['binaries'] = build['binaries']
        for p, record in report['binaries'].items():
            assert sha(ROOT / p) == record['sha256']
        report['packages_before'] = run([os.sys.executable, '-m', 'pip', 'freeze'])
        assert report['packages_before'] == build['packages_after']
        modules = {n: load(n + '_cuda', report['binaries']) for n in ('build_cost_volume', 'iou3d_nms', 'roiaware_pool3d')}
        directory = output.with_suffix('')
        directory.mkdir()
        report['cost_checks'] = [cost_check(modules['build_cost_volume'], directory, ds) for ds in (1,2)]
        report['geometry_checks'] = geometry_checks(modules['iou3d_nms'], modules['roiaware_pool3d'], directory)
        report['sources_after'] = sources(lock)
        for p, record in report['binaries'].items():
            assert sha(ROOT / p) == record['sha256']
        report['packages_after'] = run([os.sys.executable, '-m', 'pip', 'freeze'])
        assert report['packages_before'] == report['packages_after']
        report['artifacts'] = {p.name: {'sha256':sha(p), 'bytes':p.stat().st_size} for p in directory.glob('*.npz')}
        report['state'] = 'passed_native_operator_GPU_only'
    except BaseException:
        report['state'] = 'failed'
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        report['ended_unix'] = time.time()
        output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:report[k] for k in ('state','job_id')},indent=2),flush=True)


if __name__ == '__main__':
    main()
