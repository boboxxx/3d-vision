"""Build the locked native sources without installing detector dependencies."""
from pathlib import Path
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parents[3]
OPS = ROOT / 'third_party/LIGA-Stereo/liga/ops'
include = [str(p) for p in Path(sys.prefix).glob('lib/python*/site-packages/nvidia/*/include')]
specs = [
    ('build_cost_volume', ['BuildCostVolume.cpp', 'BuildCostVolume_cuda.cu'], [('WITH_CUDA', None)]),
    ('iou3d_nms', ['iou3d_cpu.cpp', 'iou3d_nms_api.cpp', 'iou3d_nms.cpp', 'iou3d_nms_kernel.cu'], []),
    ('roiaware_pool3d', ['roiaware_pool3d.cpp', 'roiaware_pool3d_kernel.cu'], []),
]
extensions = [CUDAExtension(name + '_cuda', [str(OPS / name / 'src' / s) for s in files],
                           include_dirs=include, define_macros=macros,
                           extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2']})
              for name, files, macros in specs]
setup(name='paper6-liga-operators-cu128-001', version='0.0.1', ext_modules=extensions,
      cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)})
