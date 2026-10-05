#!/usr/bin/env python3
"""Retain LIGA architecture while replacing removed runtime APIs.

Run after patch_liga.py. Full checkpoint/AP verification is still required;
source compatibility by itself does not establish reproduction.
"""
import argparse
from pathlib import Path
import re


def edit(path, transform):
    before = path.read_text(encoding="utf-8")
    after = transform(before)
    if after != before:
        path.write_text(after, encoding="utf-8")
        return str(path)


def modernize(root, mmdet):
    changed = []
    for source in (root / "liga/ops").rglob("*"):
        if source.suffix not in (".cpp", ".cu", ".h"):
            continue
        def cuda(text):
            text = text.replace("#include <THC/THC.h>", "#include <c10/cuda/CUDAException.h>")
            text = text.replace("#include <THC/THCAtomics.cuh>", "#include <ATen/cuda/Atomic.cuh>")
            text = text.replace("#include <THC/THCDeviceUtils.cuh>", "#include <ATen/ceil_div.h>")
            text = text.replace("THCCeilDiv", "at::ceil_div").replace("THCudaCheck", "C10_CUDA_CHECK")
            text = text.replace(".type().is_cuda()", ".is_cuda()")
            text = text.replace("AT_DISPATCH_FLOATING_TYPES(left.type(),", "AT_DISPATCH_FLOATING_TYPES(left.scalar_type(),")
            text = text.replace("AT_DISPATCH_FLOATING_TYPES(grad.type(),", "AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(),")
            return text.replace(".data<", ".data_ptr<")
        value = edit(source, cuda)
        if value:
            changed.append(value)
    sparse = root / "liga/models/backbones_3d_lidar/spconv_backbone.py"
    def sparse_api(text):
        text = text.replace("import spconv\n", "import spconv.pytorch as spconv\n")
        text = re.sub(r"out.features = ([^\n]+)", r"out = out.replace_feature(\1)", text)
        return text.replace("out.features += identity.features", "out = out.replace_feature(out.features + identity.features)")
    changed.append(edit(sparse, sparse_api))
    processor = root / "liga/datasets/processor/data_processor.py"
    changed.append(edit(processor, lambda t: t.replace(
        "                from spconv.utils import VoxelGenerator\n",
        "                from geocomm.compat import VoxelGenerator\n")))
    for family in ["detectors_lidar/lidar_detector3d_template.py", "detectors_stereo/stereo_detector3d_template.py"]:
        source = root / "liga/models" / family
        def weights(text):
            marker = "# GEOCOMM_SPCONV_LAYOUT"
            if marker in text:
                return text
            text = text.replace("        model_state_disk = checkpoint['model_state']", '''        # GEOCOMM_SPCONV_LAYOUT: retain every sparse kernel value.
        from geocomm.compat import adapt_spconv_state
        model_state_disk = adapt_spconv_state(self, checkpoint['model_state'])''')
            text = text.replace("self.state_dict()[key].shape == model_state_disk[key].shape",
                                "self.state_dict()[key].shape == val.shape")
            return text.replace("        self.load_state_dict(checkpoint['model_state'])", '''        from geocomm.compat import adapt_spconv_state
        self.load_state_dict(adapt_spconv_state(self, checkpoint['model_state']))''')
        changed.append(edit(source, weights))
    source = root / "liga/ops/iou3d_nms/numerical_jaccobian.py"
    changed.append(edit(source, lambda t: t.replace("from torch._six import container_abcs, istuple",
                                                   "import collections.abc as container_abcs")))
    # Same scalar/array types after NumPy removed the deprecated aliases.
    for tree in [root / "liga", mmdet / "mmdet"]:
        for source in tree.rglob("*.py"):
            value = edit(source, lambda t: re.sub(r"np\.(bool|int|float)\b", r"\1", t))
            if value:
                changed.append(value)
    init = mmdet / "mmdet/__init__.py"
    changed.append(edit(init, lambda t: t.replace("mmcv_maximum_version = '1.3'", "mmcv_maximum_version = '1.7.2'")))
    source = mmdet / "mmdet/core/post_processing/bbox_nms.py"
    changed.append(edit(source, lambda t: t.replace(
        "labels = torch.arange(num_classes, dtype=torch.long)",
        "labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)")))
    # Released inference unnecessarily executes the training-only LiDAR teacher.
    # Predictions use its outputs only for imitation loss during training.
    source = root / "liga/models/detectors_stereo/liga.py"
    changed.append(edit(source, lambda t: t if "GEOCOMM_SENSOR_ONLY_INFERENCE" in t else t.replace(
        "        for cur_module in self.module_list:\n",
        "        for cur_module in self.module_list:\n"
        "            # GEOCOMM_SENSOR_ONLY_INFERENCE: teacher is training-only.\n"
        "            if cur_module is self.lidar_model and not self.training:\n"
        "                continue\n")))
    source = root / "liga/models/dense_heads/depth_loss_head.py"
    changed.append(edit(source, lambda t: t if "GEOCOMM_OPTIONAL_DEPTH_DIAGNOSTICS" in t else t.replace(
        "    def forward(self, batch_dict):\n",
        "    def forward(self, batch_dict):\n"
        "        # GEOCOMM_OPTIONAL_DEPTH_DIAGNOSTICS: no GT needed to predict.\n"
        "        if not self.training and 'depth_gt_img' not in batch_dict:\n"
        "            return batch_dict\n")))
    source = root / "tools/test.py"
    changed.append(edit(source, lambda t: t.replace("np.random.seed(1024)",
        "np.random.seed(int(os.environ.get('GEOCOMM_RANK_SEED', '1024')))")))
    for name in ["train.py", "test.py"]:
        source = root / "tools" / name
        changed.append(edit(source, lambda t: t.replace(
            "parser.add_argument('--local_rank', type=int, default=0,",
            "parser.add_argument('--local-rank', '--local_rank', dest='local_rank', type=int, default=0,")))
    return [value for value in changed if value]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--mmdet", type=Path, required=True)
    args = p.parse_args()
    print("\n".join(modernize(args.root, args.mmdet)) or "Already modernized")
