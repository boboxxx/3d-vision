# SRCNN author core recovery and CPU mathematical translation

Lock before any SRCNN forward. Read-only recovery has identified the original
author project archives, rather than a third-party reimplementation:
https://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html . TestSRCNN_v1.zip SHA256
bfa68ca613c1326a59e0c34353205a254ab2b67e34df7f04e28eef567980af30;
trainingSRCNN_train.zip
001146419f7acfb12a3e7929c8acd5de88a08d687d6881085f81321ad6982b1a.
All archive members were inspected for path traversal before extraction.

The test release contains eight luminance models:9-1-5/91-image x2/x3/x4,
9-1-5/ImageNet x3,9-3-5/ImageNet x3 and9-5-5/ImageNet x2/x3/x4.
No selection between them. Inspect all six learned arrays in every MAT file;
record full source/model hashes and shapes. SRCNN.m uses same-size replicate
boundary correlation, two ReLUs and final linear luminance output. The demo
uses YCbCr luminance, integer2/3/4 bicubic down/up sampling, modcrop and border
shaving. It does not define Cao2025's10/30/50×stereo compression interface.

Implement a CPU-only mathematical translation of the unchanged SRCNN.m core,
MATLAB-column-major filter interpretation, fixed learned values and replicate
boundaries. Six readonly tensors per model. Use double input and explicit
float64 arithmetic to compare PyTorch correlation to an independent SciPy
nearest-boundary per-channel correlation plus literal source-order sums.
This tests the mathematical double-input function. No MATLAB runtime, released
demo's mixed single/double numerics or full preprocessing equivalence is claimed.

Before running, fixed synthetic16×20inputs:allzeros;unitimpulse atrow7/column11;
NumPy PCG64 seed20261004 uniform[0,1). All8models×3inputs=24comparisons, every
native output pixel and all six source-loaded readonly tensors. Require finite
outputs/native geometry and maxabsolute error<=1e-10. No KITTI, GT, calibration,
model optimization, source bitstream, PSNR/SSIM, radio or detector AP inputs.
Record source/runtime/weight identities before and after; unique outputs only.

This recovers a trustworthy author core for the required comparator. A separate
declared compression/color/resizing/quantization/operating-point protocol and
training/native inference are still required. Do not call one x3 checkpoint
the original10/30/50baseline, silently resize/crop the stereo detector geometry,
or substitute this synthetic gate for complete SRCNN baseline results.
