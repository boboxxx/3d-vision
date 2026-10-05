# Artemis complete detector framework engineering, 001

Locked before package resolution, installation and compilation on 2026-10-05.
This continues the actual RTX PRO6000 native-operator gate; it does not train
a detector or establish AP, sparse CUDA compatibility or geometry utility.

Preserve the existing Python3.12 / torch2.7.1+cu128 runtime, frozen LIGA and
mmdetection sources, CUDA12.8 toolkit and three native operator binaries.
Install additional packages only under
`dependencies/artemis-framework-overlay-001/site-packages`. Use the matching
official torchvision0.22.1+cu128 wheel. Resolve pinned top-level requirements
against the actual existing package freeze; save the entire pip dry-run report
and exact official archive URLs/SHA256. Reject any resolution that replaces
an existing package, Torch, NVIDIA runtime or NumPy. Commit the resolved input
lock before installing. Unpinned transitive packages are engineering dependencies,
not a model selection, and become exact locked artifacts before execution.

Compile unchanged MMCV-full1.7.2 from its official PyPI source archive, SHA256
aa276a2c68fa84db9bbcdcf6ae941e30f0cb4a7f175bd283516669e00bab3759.
Retain source/archive/build output and any failure. A short CPU Slurm job uses
8 CPUs, 32GiB, two hours, GNU12, at most four compiler workers, explicit sm120,
FORCE_CUDA and the existing toolkit. Do not compile on the login node. CPU-only
spconv2.3.8/cumm0.7.11 wheels may satisfy registry imports; they cannot establish
GPU sparse convolution or teacher training. No fake modules or substituted
operators. If the original framework is incompatible, retain the failed run
and lock a narrowly scoped repair before retrying changed code.

Success of this stage means all dependency archives match the lock, MMCV
compiles, real mmcv/mmdet/liga registry imports succeed, Torch/NumPy still load
from the original runtime, and its package freeze and all frozen input hashes
are unchanged before/after. Require actual Slurm terminal exit0 separately.
A later locked sensor-only KITTI forward on actual RTX PRO6000 must load all
484 author states strictly, prohibit teacher/GT/loss calls, preserve weights,
and record actual outputs before this is called a usable native detector.

Primary references checked on 2026-10-05:
[PyTorch matching versions](https://pytorch.org/get-started/previous-versions/),
[spconv installation and CPU-wheel limitations](https://github.com/traveller59/spconv),
[MMCV build instructions](https://mmcv.readthedocs.io/en/master/get_started/build.html).
These describe installation routes; compatibility of this combination is
unverified until the actual build and task run.
