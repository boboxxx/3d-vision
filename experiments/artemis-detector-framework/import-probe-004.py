"""Real framework imports using existing native binaries; no GPU/task forward."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OVERLAY = ROOT / 'dependencies/artemis-framework-overlay-004/site-packages'
import torch
import numpy
assert torch.__version__ == '2.7.1+cu128' and numpy.__version__ == '1.26.4'
assert 'envs/torch-cu128-001/' in torch.__file__ and 'envs/torch-cu128-001/' in numpy.__file__
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'third_party/LIGA-Stereo'),
               str(ROOT / 'third_party/mmdetection_kitti')]
build = json.loads((ROOT / 'data/engineering/artemis-detector-operators-build-001.json').read_text())
for package, name in [('build_cost_volume', 'build_cost_volume_cuda'),
                      ('iou3d_nms', 'iou3d_nms_cuda'), ('roiaware_pool3d', 'roiaware_pool3d_cuda')]:
    paths = [ROOT / p for p in build['binaries'] if Path(p).name.startswith(name + '.')]
    assert len(paths) == 1
    full = 'liga.ops.' + package + '.' + name
    spec = importlib.util.spec_from_file_location(full, paths[0])
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
import mmcv
import mmcv.ops
import torchvision
import spconv.pytorch
import mmdet.models
import liga.models
assert mmcv.__version__ == '1.7.2' and Path(mmcv.__file__).is_relative_to(OVERLAY)
assert torchvision.__version__ == '0.22.1+cu128' and Path(torchvision.__file__).is_relative_to(OVERLAY)
assert len(liga.models.build_network.__name__) > 0
print(json.dumps(dict(state='passed_real_framework_registry_imports_only',
    torch_version=torch.__version__, numpy_version=numpy.__version__,
    mmcv_version=mmcv.__version__, torchvision_version=torchvision.__version__,
    torch_path=torch.__file__, numpy_path=numpy.__file__, mmcv_path=mmcv.__file__,
    sparse_GPU_unverified=True, detector_forward_unverified=True,
    CUDA_initialized=torch.cuda.is_initialized()), indent=2))
assert not torch.cuda.is_initialized()
