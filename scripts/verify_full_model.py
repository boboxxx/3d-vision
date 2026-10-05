#!/usr/bin/env python3
"""Strict full LIGA checkpoint audit and optional full-resolution synthetic pass.

Only dataset geometry is stubbed; every upstream model module is built. Synthetic
images are engineering input, so this command never computes or claims KITTI AP.
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
LIGA = ROOT / "third_party/LIGA-Stereo"
sys.path[:0] = [str(ROOT / "src"), str(LIGA), str(ROOT / "third_party/mmdetection_kitti")]
from geocomm.compat import adapt_spconv_state
from geocomm.evidence import sha256, serializable


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=LIGA / "configs/stereo/kitti_models/liga.3d-and-bev.yaml")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--calibration", type=Path)
    p.add_argument("--forward", action="store_true")
    p.add_argument("--codec-gradient", action="store_true",
                   help="synthetic full training-loss backward with detector frozen; no optimizer step")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.config = args.config.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        p.error("retain previous verification outputs; use a unique output")
    if (args.forward or args.codec_gradient) and args.calibration is None:
        p.error("forward/gradient checks require --calibration")
    args.calibration = args.calibration.resolve() if args.calibration else None
    report = dict(evidence_type="engineering_only", state="running",
                  checkpoint_sha256=sha256(args.checkpoint), config_sha256=sha256(args.config),
                  started_at_unix=time.time(), torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
    torch.manual_seed(17)
    np.random.seed(17)
    os.chdir(LIGA)
    try:
        from easydict import EasyDict
        from liga.config import cfg_from_yaml_file
        from liga.models import build_network
        config = cfg_from_yaml_file(str(args.config), EasyDict())
        data = config.DATA_CONFIG
        # Match DatasetTemplate's float32 geometry rather than NumPy defaults.
        pc = np.asarray(data.POINT_CLOUD_RANGE, dtype=np.float32)
        voxel = np.asarray(data.VOXEL_SIZE, dtype=np.float32)
        stereo_voxel = np.asarray(data.STEREO_VOXEL_SIZE, dtype=np.float32)
        dataset = SimpleNamespace(class_names=config.CLASS_NAMES, point_cloud_range=pc,
                                  voxel_size=voxel, grid_size=np.round((pc[3:] - pc[:3])/voxel).astype(int),
                                  stereo_voxel_size=stereo_voxel,
                                  stereo_grid_size=np.round((pc[3:] - pc[:3])/stereo_voxel).astype(int),
                                  boxes_gt_in_cam2_view=data.BOXES_GT_IN_CAM2_VIEW,
                                  point_feature_encoder=SimpleNamespace(num_point_features=3))
        with tempfile.TemporaryDirectory() as directory:
            dist.init_process_group("gloo", init_method="file://" + directory + "/process_group", rank=0, world_size=1)
            model = build_network(config.MODEL, len(config.CLASS_NAMES), dataset).cuda().eval()
            checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            state = adapt_spconv_state(model, checkpoint["model_state"])
            expected = model.state_dict()
            missing = [key for key, value in expected.items()
                       if "semantic_link" not in key and (key not in state or state[key].shape != value.shape)]
            unexpected = [key for key in state if key not in expected]
            report.update(expected_state_tensors=len(expected), checkpoint_state_tensors=len(state),
                          missing_detector_weights=missing, unexpected_weights=unexpected,
                          checkpoint_epoch=checkpoint.get("epoch"), checkpoint_version=checkpoint.get("version"))
            report["random_untrained_link_tensors"] = [key for key in expected
                if "semantic_link" in key and key not in state]
            if missing or unexpected:
                raise RuntimeError("incomplete full detector checkpoint: " + repr((missing, unexpected)))
            weights = dict(expected)
            weights.update(state)
            model.load_state_dict(weights, strict=True)
            report["modules"] = [name for name, _ in model.named_children()]
            report["parameters"] = sum(v.numel() for v in model.parameters())
            print("Full checkpoint matched every detector and LiDAR teacher tensor", flush=True)
            if args.forward:
                from liga.utils.calibration_kitti import Calibration
                calib = Calibration(str(args.calibration))
                calib.offset(0, 55)
                batch = dict(batch_size=1, left_img=torch.randn(1, 3, 320, 1280, device="cuda"),
                             right_img=torch.randn(1, 3, 320, 1280, device="cuda"),
                             calib=[calib], image_shape=np.array([[320, 1280]]))
                teacher_calls = []
                hook = model.lidar_model.register_forward_hook(lambda *unused: teacher_calls.append(1))
                feature_teacher_calls = []
                student_enabled = getattr(model.backbone_3d,'student_semantic_link_encoder',None) is not None
                feature_hooks = [module.register_forward_hook(lambda *unused: feature_teacher_calls.append(1))
                    for module in [model.backbone_3d.feature_backbone,model.backbone_3d.feature_neck]] if student_enabled else []
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.no_grad():
                    predictions, diagnostics = model(batch)
                torch.cuda.synchronize()
                hook.remove()
                for feature_hook in feature_hooks:
                    feature_hook.remove()
                if teacher_calls:
                    raise RuntimeError("LiDAR teacher was executed during sensor-only inference")
                if feature_teacher_calls:
                    raise RuntimeError('original feature teacher executed during student inference')
                for key in ["pred_boxes", "pred_scores"]:
                    if not torch.isfinite(predictions[0][key]).all():
                        raise RuntimeError("nonfinite " + key)
                report["synthetic_forward"] = dict(input_shape=[1, 3, 320, 1280],
                    elapsed_seconds=time.perf_counter() - start,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    predictions=int(predictions[0]["pred_boxes"].shape[0]),
                    inference_inputs="random stereo RGB and calibration; no LiDAR or ground truth",
                    teacher_forward_calls=len(teacher_calls),
                    student_enabled=student_enabled,feature_teacher_forward_calls=len(feature_teacher_calls),
                    communication_accounting=batch.get("communication_accounting"),
                    timing_note="single cold engineering pass; not a latency benchmark")
                print("Full-resolution synthetic sensor-only inference passed", flush=True)
                del predictions, diagnostics, batch
                torch.cuda.empty_cache()
            if args.codec_gradient:
                from liga.utils.calibration_kitti import Calibration
                from geocomm.compat import VoxelGenerator
                if model.backbone_3d.semantic_link is None and getattr(model,'rgb_semantic_link',None) is None:
                    raise RuntimeError("gradient check requires enabled semantic link")
                model.train()
                for name, param in model.named_parameters():
                    param.requires_grad_("semantic_link" in name and "generic_score" not in name)
                for module in model.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        module.eval()
                calib = Calibration(str(args.calibration))
                calib.offset(0, 55)
                points = np.random.uniform(pc[:3], pc[3:], size=(4000, 3)).astype(np.float32)
                voxels, coordinates, counts = VoxelGenerator(voxel.tolist(), pc.tolist(), 5, 40000).generate(points)
                box = torch.tensor([[[20., 0., -.5, 3.9, 1.6, 1.56, 0., 1.]]], device="cuda")
                depth = torch.zeros(1, 1, 320, 1280, device="cuda")
                depth[:, :, 80:240:8, 300:1000:8] = 20.
                batch = dict(batch_size=1, left_img=torch.randn(1, 3, 320, 1280, device="cuda"),
                             right_img=torch.randn(1, 3, 320, 1280, device="cuda"),
                             calib=[calib], image_shape=np.array([[320, 1280]]),
                             voxels=torch.from_numpy(voxels).cuda(),
                             voxel_coords=torch.from_numpy(np.pad(coordinates, ((0, 0), (1, 0)))).cuda().int(),
                             voxel_num_points=torch.from_numpy(counts).cuda().long(),
                             gt_boxes=box, gt_boxes_no3daug=box.clone(),
                             gt_boxes_2d=torch.tensor([[[550., 100., 700., 200.]]], device="cuda"),
                             gt_centers_2d=torch.tensor([[[625., 150.]]], device="cuda"), depth_gt_img=depth)
                torch.cuda.reset_peak_memory_stats()
                losses, scalars, _ = model(batch)
                loss = losses["loss"].mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite synthetic training loss")
                loss.backward()
                gradients = {}
                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue
                    if param.grad is None or not torch.isfinite(param.grad).all():
                        raise RuntimeError("missing/nonfinite task gradient: " + name)
                    gradients[name] = float(param.grad.norm())
                if not gradients or max(gradients.values()) == 0:
                    raise RuntimeError("zero codec gradients")
                report["synthetic_codec_backward"] = dict(loss=float(loss.detach()),
                    loss_components=scalars, gradient_norms=gradients,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(), optimizer_steps=0,
                    input_note="synthetic RGB, point cloud and labels; validates full loss integration only")
                print("Full detector task losses reached codec parameters with finite gradients", flush=True)
            dist.destroy_process_group()
        report["state"] = "passed"
    except BaseException as exc:
        report["state"] = "failed"
        report["exception"] = repr(exc)
        if dist.is_initialized():
            dist.destroy_process_group()
        raise
    finally:
        report["ended_at_unix"] = time.time()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=serializable), encoding="utf-8")


if __name__ == "__main__":
    main()
