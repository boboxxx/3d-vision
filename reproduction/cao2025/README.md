# Cao2025 communication reproduction workspace

This workspace is isolated from the direct-detection experiment code and
does not change its student, receiver, dataset or training budget.

Primary specification: [Cao et al., Sections III–IV and Appendix A](https://arxiv.org/pdf/2502.12735).
The PDF figures6–7 were visually checked. The appendix/prose cannot define a
single executable architecture: inverse pixel shuffle yields144 channels but
the next layer accepts256; the described recovery concatenation and channel
table disagree; parameter sharing is incompletely specified. These remain
author-specification ambiguities, rather than implementation permission to
omit the optical-flow or full residual recovery networks.

The explicit exploratory variant here uses global downsampling6 and key2,
changes the incompatible global final convolution to144 inputs, adds an explicit
67→64 recovery bridge for warped64-channel features plus received3-channel
global data, and keeps independent view-specific recovery/fusion parameters.
Every recovery stack contains all30 residual blocks. Actual parameter counts
must be recorded; they need not match the paper's incompletely specified counts.
The code is a **documented reproduction variant**, not the located author codec
or a reproduced published AP result. Padding is bottom/right to multiples6;
outputs are cropped back without shifting calibration coordinates.

SpyNet/flow-warp definitions are extracted from the unchanged official BasicSR
files pinned at8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a. Apache2 license is
retained. Only package imports and the registry decorator are removed from the
AST; original class/function bodies execute unchanged. Strict supplied weight
loading is required; no random flow network is substituted. This is a practical
SpyNet source, not evidence that Cao used the exact same released weights.

The [VRT author release](https://github.com/JingyunLiang/VRT/releases/tag/v0.0)
supplies `spynet_sintel_final-3d2a1287.pth`; identity is recorded in
`upstream/checkpoint.json`. All60 learned tensors exactly match the loaded model.
The released container omits fixed `mean/std`; upstream itself loads before
registering these buffers. Our loader supplies only those constructor constants
and requires a strict complete state match. A missing learned tensor is rejected.

The local CPU engineering probe passed: real pretrained flow, positive-x warp
sampling direction, padded197×203→198×204→197×203 stereo output, all638 parameter
tensor gradients finite, all five major component groups with nonzero gradients,
and stage1 global-only warmup. This independent-branch variant contains
12,349,974 parameters. These synthetic untrained checks do not establish RGB
quality, wireless performance or detection AP. Run `probe.py --checkpoint PATH
--output JSON` inside this directory to repeat them.

Remaining work includes equal physical-channel budgets, the complete staged
training schedule and final Stereo-RCNN inference/AP audit.
Do not label the current architecture or synthetic checks a full reproduction.

`extract_rois.py` now executes official frozen YOLOv5n v7.0 in a standalone
CPU process on sheng. The documented variant uses COCO car/bus/truck classes,
confidence0.25 and NMS IoU0.45, with official letterbox/coordinate scaling and
rounded native-image boxes. Source/weights are pinned in `upstream/yolov5.json`.
Only the RGB PNGs and train-only fold IDs are read. Three real stereo pairs
passed; this is not an ROI detection-quality or original-threshold reproduction.
Thresholds/classes remain fixed for full-fold extraction. Each record includes
image hashes, boxes/confidence/classes, original/input shapes and the union-mask
area/hash. Full-fold records require an independent coverage/mask audit.

Both full folds (3340 train/372 holdout) now have independently audited ROI
caches. The [wireless variant](wire-protocol.md) includes TableIX CNNs, sparse
key transmission, serialized/protected box control and actual receiver parsing.
The local CPU probe passes with12,970,798 total parameters and718 finite
parameter gradients. CRC corruption and missing data are rejected; energy and
channel variance are checked. These are untrained engineering checks. No
complete staged wireless training or reproduced original detection AP exists.
