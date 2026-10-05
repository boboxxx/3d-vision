# Artemis native detector operators, engineering 001

Locked before execution on 2026-10-04. This is an engineering prerequisite for the geometry-risk pilot and full native detector on RTX PRO 6000, with zero detector training or AP claims.

Use the existing project Python 3.12.8 / PyTorch 2.7.1+cu128 runtime, unchanged. Install only project-local NVIDIA redistributable compiler, runtime headers/libraries and CCCL from the official CUDA 12.8.1 JSON. Exact metadata, component sizes/SHA256 and every native C++/CUDA input are locked in `inputs-001.json`. No system toolkit, driver, shared environment or sheng source changes. Compilation targets sm_120 explicitly, with at most four CPU compile workers in a short CPU Slurm allocation (8 CPUs, 32 GiB, two hours). Retain archives, licenses, all build logs and failure reports.

Compile unchanged modernized LIGA native sources for build_cost_volume, iou3d_nms and roiaware_pool3d into an independent build directory. The standalone extension build bypasses detector Python dependencies; this establishes operator availability only. It does not establish compatibility of MMCV, mmdetection, spconv or the full detector.

Only after successful compilation, allocate enginf / gpu:RTX:1 and verify the actual device is RTX PRO 6000 Blackwell, capability (12,0). Release at job exit. Compare native cost-volume forward and both feature gradients against independent CPU PyTorch interpolation for shifts 0, 1 and 1.5, including invalid left boundaries and downsample 1/2, FP32 tolerances 2e-5. Check analytic axis-aligned BEV overlap/IoU, NMS duplicate removal and separate-box retention; compare ROI membership against explicit coordinate bounds on both CPU and GPU. Save actual fixture arrays, gradients, outputs, binary SHA256, device/runtime identities and source before/after hashes. No optimizer or weight file used.

Success requires finite matching values, all frozen inputs unchanged, project runtime package freeze unchanged, and independently verified actual Slurm terminal exit 0. A build-only success is not GPU execution. An operator probe is not a full native detector, downstream loss or AP experiment. Preserve any failed attempt and lock a repair before rerunning changed sources or checks.

Official installation reference: https://docs.nvidia.com/cuda/archive/12.8.1/cuda-installation-guide-linux/index.html ; metadata: https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json . Component licenses are retained with their archives.
