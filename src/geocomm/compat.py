"""Adapters for released LIGA mathematical operations on modern runtimes."""
import numpy as np


class VoxelGenerator:
    """spconv 2 CPU voxelizer with the legacy return contract and XYZ inputs."""
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        self.parameters = dict(vsize_xyz=voxel_size,
                               coors_range_xyz=point_cloud_range,
                               max_num_points_per_voxel=max_num_points,
                               max_num_voxels=max_voxels)
        self.generator = None

    def generate(self, points):
        from spconv.utils import Point2VoxelCPU3d
        from cumm import tensorview as tv
        if self.generator is None:
            self.generator = Point2VoxelCPU3d(num_point_features=points.shape[1], **self.parameters)
        voxels, coords, counts = self.generator.point_to_voxel(tv.from_numpy(np.ascontiguousarray(points)))
        # Voxelizer storage is reused; data-loader samples must own their arrays.
        return voxels.numpy().copy(), coords.numpy().copy(), counts.numpy().copy()


def adapt_spconv_state(model, state):
    """Translate only identified spconv 1 RSCK kernels to spconv 2 KRSC.

    Dense 3D convolution weights are deliberately excluded. Spatial kernel
    order is unchanged. Never silently drop an incompatible sparse kernel.
    """
    output = dict(state)
    for name, module in model.named_modules():
        if not type(module).__module__.startswith("spconv.pytorch"):
            continue
        weight = getattr(module, "weight", None)
        key = name + ".weight" if name else "weight"
        if weight is None or weight.ndim != 5 or key not in output:
            continue
        old = output[key]
        if old.shape == weight.shape:
            continue
        converted = old.permute(4, 0, 1, 2, 3).contiguous()
        if converted.shape != weight.shape:
            raise ValueError("unsupported sparse weight layout: " + key)
        output[key] = converted
    return output
