#!/usr/bin/env python3
"""Modernize the pinned official 1.0 branch without replacing its operators."""
import argparse
from pathlib import Path
import re


def modernize(root):
    changed = []
    for path in (root / 'lib/model/csrc').rglob('*'):
        if path.suffix not in {'.cpp', '.cu', '.h'}:
            continue
        before = path.read_text()
        text = before.replace('#include <THC/THC.h>', '#include <c10/cuda/CUDAException.h>')
        text = text.replace('#include <THC/THCAtomics.cuh>', '#include <ATen/cuda/Atomic.cuh>')
        # ceil_div must also be callable inside the original device NMS kernel.
        text = text.replace('#include <THC/THCDeviceUtils.cuh>',
                            'template <typename T> __host__ __device__ T stereo_ceil_div(T a, T b) { return (a + b - 1) / b; }')
        text = text.replace('THCCeilDiv', 'stereo_ceil_div').replace('THCudaCheck', 'C10_CUDA_CHECK')
        text = text.replace('.type().is_cuda()', '.is_cuda()').replace('.data<', '.data_ptr<')
        text = re.sub(r'AT_DISPATCH_FLOATING_TYPES\((\w+)\.type\(\),',
                      r'AT_DISPATCH_FLOATING_TYPES(\1.scalar_type(),', text)
        text = text.replace('dets.type() == scores.type()',
                            'dets.scalar_type() == scores.scalar_type()')
        if path.name == 'nms.cu' and 'THCState *state' in text:
            start = text.index('  THCState *state')
            end = text.index('  dim3 blocks', start)
            text = text[:start] + '''  auto mask_tensor = at::empty({boxes_num * col_blocks}, boxes.options().dtype(at::kLong));
  auto mask_dev = reinterpret_cast<unsigned long long*>(mask_tensor.data_ptr<int64_t>());

''' + text[end:]
            text = text.replace('nms_kernel<<<blocks, threads>>>',
                                'nms_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>')
            text = text.replace('  THCudaFree(state, mask_dev);\n', '')
        if before != text:
            path.write_text(text)
            changed.append(str(path.relative_to(root)))
    for path in (root / 'lib').rglob('*.py'):
        before = path.read_text()
        text = re.sub(r'np\.(float|int|bool)\b', r'\1', before)
        text = text.replace('scipy.array(', 'np.array(')
        text = text.replace('yaml.load(f)', 'yaml.safe_load(f)')
        # Upstream normalizes pixels with (W-1,H-1), matching align_corners=True.
        text = text.replace("padding_mode='border')", "padding_mode='border', align_corners=True)")
        if before != text:
            path.write_text(text)
            changed.append(str(path.relative_to(root)))
    return changed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('third_party/Stereo-RCNN'))
    args = parser.parse_args()
    print('\n'.join(modernize(args.root)))
