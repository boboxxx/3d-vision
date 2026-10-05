# Complete framework integration: real registry imports now closed

Unchanged MMCV-full1.7.2 compiles for sm120 and real mmcv.ops, torchvision,
spconv.pytorch, mmdet.models and liga.models imports pass on Artemis. Import006
Slurm11424824 actually ends COMPLETED0:0. All426 locked inputs and the original
Python3.12/torch2.7.1+cu128 environment remain unchanged. This closes CPU
compilation/registry compatibility. Complete native GPU forward and frozen
detection-loss input backward subsequently execute successfully; see analysis002.
Sparse GPU LiDAR teacher and complete numerical forward acceptance remain unverified.

Build005's unchanged MMCV wheel SHA is
2539dbc0811b5478a80db0f671d404f30358884ad1f22f1d56c09b1b587587a6.
It uses the same48 verified official archives and independent overlay005.
The prior CUDA-header failure was repaired only by including the existing
NVIDIA math-header directories; the compile-only ATen/CUDAContext probe passed.
No toolkit math libraries, base packages or original operators were replaced.
CPU spconv/cumm wheels establish registry imports, not sparse GPU convolution.

Build005 actually ended FAILED1:0 at real import: LIGA's author-generated
version.py was absent. The compiler had succeeded. Protocol008 retained all
complete logs and reused the installed overlay. Import006 generated only
__version__="0.1.0+0000000", exactly the author's setup.py rule for a checkout
without .git, already used on sheng. No frozen source was edited. The same
unchanged real import-probe005 then passed, without GPU initialization.

Earlier failures remain: fire0.7.0 lacked a wheel; the official torchvision
index delegated to download-r2;330 frozen Python sources were absent, with27
existing identical and no mismatches; only missing files were staged. CDN403
occurred on login and compute; the exact URL returned206 to the pip client
identifier, disproving a compute-only restriction. All48 unchanged official
archives184,354,145 bytes were fully hashed. These operational failures do not
measure model quality.

Evidence: [build005 report](/Users/chen/Documents/ChatGPT/paper6/data/engineering/artemis-framework-build-005.json),
[complete build logs](/Users/chen/Documents/ChatGPT/paper6/logs/artemis-framework-build-005/mmcv-build.log),
[real import006](/Users/chen/Documents/ChatGPT/paper6/data/engineering/artemis-framework-import-006.json),
[actual terminal](/Users/chen/Documents/ChatGPT/paper6/data/engineering/artemis-framework-import-006-terminal.json),
[version-only prelock](/Users/chen/Documents/ChatGPT/paper6/experiments/artemis-detector-framework/protocol-008.md).
