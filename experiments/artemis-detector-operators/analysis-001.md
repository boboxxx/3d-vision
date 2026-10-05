# Actual RTX PRO6000 native operator execution closed

Protocol e5d7094 and immutable native-source/compiler inputs precede CPU build
11423973 and GPU probe11423980. The latter actually executes on
artemis-rtx-03, NVIDIA RTX PRO6000 Blackwell Server Edition, capability12.0,
PyTorch2.7.1+cu128/CUDA12.8, driver610.57.04,97887MiB device capacity.
Slurm parent/batch/extern/task all finish COMPLETED/0:0; GPU allocation is
released. It is no longer pending(QOSGrpGRES).

Native cost-volume fixtures exercise shifts0,1,1.5, left invalid boundaries,
downsample1/2, forward and both feature gradients. Native-versus-independent
CPU reference maximum errors are0 for forwards and<=5.97e-8 for gradients.
All1104 actual output/gradient float32 values replay independently in local
NumPy scalar loops with maximum difference5.97e-8. Actual array hashes,
saved expected values and every report error match. Axis-aligned BEV overlap,
IoU CPU/GPU, NMS keep[0,2,3] and CPU/GPU point-in-box memberships independently
agree with analytic rectangle/coordinate bounds locally. Points are away from
surfaces and all tested box yaw angles are0; rotated/boundary conventions
and ROI feature pooling are not validated by this fixture.

Fresh native terminal closure verifies all10 C++/CUDA source hashes, three
binary identities, execution scripts, input lock, build report and current
package freeze against before/after probe. Complete local verification covers
three actual NPZ fixtures, GPU report, stdout/stderr and terminal evidence,
seven artifacts. Native binaries are checked on Artemis, not executed on Mac.
No native/source/runtime repair or new GPU dispatch was required.

This establishes the declared isolated operator prerequisite. Full detector
imports, MMCV/mmdetection/spconv integration, native task forward/backward,
full memory budget, teacher posterior extraction, training and AP remain
unverified on this runtime. Continue with an isolated full-framework engineering
gate before allocating it to a research comparison. The completed55-second
Slurm job is not a full training experiment.

Evidence: `data/engineering/artemis-detector-operators-GPU-001.json`,
`data/engineering/artemis-detector-operators-GPU-terminal-001.json`,
`data/provenance/artemis-detector-operators-GPU-local-verification-001.json`.
