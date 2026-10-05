# Public validation ROI and source-quality execution prelock

Lock before public-validation ROI extraction, PSNR/SSIM implementation or
quality inspection. Existing source parameters, weights and AP protocols stay
fixed; no ROI, quality or AP result selects a compressor. Clean detector and
F9 results, complete JPEG/JP2 byte ratios and SRCNN synthetic interface checks
were previously inspected. Cao's exact YOLO weights/thresholds and printed
global/key definitions are unknown; these are disclosed measurement choices.

## Sensor-only fixed ROI

Reuse the already pinned YOLOv5n v7.0 revision
915bbf294bb74c859f0b41f1c23bc395014ea679 and checkpoint SHA256
4f180cf23ba0717ada0badd6c685026d73d48f184d00fc159c2641284b2ac0a3.
Use the existing ROI variant without changing its network or extraction
operation: official letterbox640, stride auto, FP32 CPU, COCO car/bus/truck
IDs2/5/7, confidence0.25, class-aware NMS IoU0.45, max_det300, official
scale_boxes followed by rounding to native integer coordinates. CPU threads2.
Apply independently to left/right clean native RGB sensor images; no labels,
calibration, LiDAR, received RGB, compressor or detector predicts the masks.
Models stay eval, no gradients/optimizer; all initial and terminal parameter/
buffer identities, actual source tree/checkpoint and PNG hashes are retained.

Engineering uses only train000000/000003, four views. Native complete image/
box/mask audit and local four-view metadata verification precede main IDs.
Main uses all3769 public val IDs in exact split order/SHA256
657ac4bcc1e156e5b106a4ca18e1f88e012787ea1d2b5d0adeea97fee903fa86,
7538views. No subset, label-based boxes or missing-view drop. Independent
auditor reopens every native PNG and reconstructs each saved union mask from
integer half-open rectangles [x1,x2)×[y1,y2). Overlaps count once. Record empty
views; their key quality is undefined rather than zero/perfect. Source PNG
and all implementation/model identities must remain unchanged after extraction.
Local verification checks all7538 actual saved metadata/mask identities and
complete native audit. It does not claim local YOLO or raw PNG replay.

These ROI measurements have no physical channel. Do not add them to delivered
source bytes unless a later wire protocol actually carries their geometry.
ROI is a separate saved sender-side definition used identically across source
conditions. It never enters current JPEG/JP2 sender or detector inference.
Use a unique CPU-only native cycle outside all frozen detector/source roots;
do not interrupt/compete for GPU with the active twelve main AP endpoints.

## Image-quality convention

For each received native RGB HWC array and corresponding clean RGB8/255,
record global g and key k as the nonoverlapping ROI union. Native geometry
must match exactly. Never modcrop, detector-crop, resize or shave boundaries.
For integer decoded JPEG/JP2, normalize RGB8 by255. For SRCNN and original-
inspired neural RGB, the original-style quality view is
round-to-nearest-even(255*clip(RGB,0,1))/255. For ECSIC retain clipped floating
RGB without the extra8-bit rounding, disclosing the original's exception.
Also report unclipped float quality as a separately named extension wherever
a received float array exists. These quality conversions never alter stored
received-task tensors or the already fixed unclipped detector input.

Use float64 mean squared RGB error over all selected channel values.
PSNR=10log10(1/MSE). Zero MSE is explicit perfect equality, serialized as
psnr_infinite=true and psnr_dB=null rather than a nonfinite JSON number.
Empty ROI has mse/psnr/ssim null and undefined_empty_ROI=true.

SSIM uses the original Wang-style local formula independently per RGB channel:
11×11 Gaussian with sigma1.5, normalized separable weights, reflect padding,
population weighted variance/covariance (no Bessel correction), C1=.01²,
C2=.03² and L=1. The formula is
((2*mu_x*mu_y+C1)*(2*cov_xy+C2))/
((mu_x²+mu_y²+C1)*(var_x+var_y+C2)). Average channel SSIM maps over full native
pixel centers for g; for k average only centers inside the union mask. Local
windows may include background at ROI boundaries, explicitly disclosed;
do not calculate isolated resized crops or weight overlapping boxes twice.
No SSIM clamping. Images smaller than11px are engineering fixtures only.

Retain every view's selected pixel count, squared-error sum, SSIM sum/count,
ROI/image/received identities, quantization policy and out-of-range fraction.
Primary aggregate is pooled MSE and its PSNR, and pooled SSIM centers, across
both views/all frames. Also disclose arithmetic means of finite per-view
PSNR as secondary, with perfect/empty counts; no silent log averaging or
omission of erasures. Source-only conditions presently have no erasures;
future channel erasure quality is undefined, with outage and delivered-only
quality separately labeled and all attempts kept in detection denominator.

Before native complete conditions, CPU fixtures must independently verify
identity/constant/impulse SSIM including reflect boundaries, ROI overlap/empty
handling, pooled versus per-view PSNR, quantization half ties and overshoot,
and dtype/geometry rejection. Native engineering covers all12 already closed
JPEG/JP2 training pairs. Fresh complete source/ROI/cache identity and numerical
quality audit, then local all-record aggregate recomputation, precede result
release. Report all six JPEG/JP2 conditions, and later all original sources,
without changing codec rates or picking favorable quality conventions.
