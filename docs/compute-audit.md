# Full-model compute audit, 2026-10-03

The optional student sender reduces executed dense convolution/linear compute,
while its measured transmitter time remains higher than the RGB encoder.
No codec is trained and these measurements contain no KITTI accuracy evidence.

| Complete pipeline | Tx dense GFLOPs lower bound | Rx dense GFLOPs lower bound | Median wall ms | Median Tx stream ms |
|---|---:|---:|---:|---:|
| Clean LIGA | — | 1393.67 (whole detector) | 90.95 | — |
| Processed-cost geometry | 922.27 | 473.39 | 101.53 | 42.73 |
| Raw-cost geometry | 619.01 | 781.74 | 102.18 | 30.22 |
| RGB reconstruction | 15.37 | 1489.75 | 94.48 | 0.74 |
| Raw-cost geometry, nonlinear posterior | 622.61 | 781.74 | 104.03 | 32.05 |
| Student raw-cost, nonlinear posterior | 12.48 | 781.74 | 88.69 | 15.99 |
| Student + task sensitivity | 12.51 | 781.74 | 89.69 | 17.11 |

Evidence: `data/engineering/full-compute-profile-001.json` and
`raw-compute-profile-001.json`, and `coupled-posterior-compute-profile-001.json`,
with corresponding original logs. The first raw-cost row uses the earlier
linear posterior and is retained as historical engineering evidence.
RTX 4090, torch 2.5.0+cu118, one fixed synthetic stereo pair of shape
1x3x320x1280 per view, author detector checkpoint, three warm-ups and twenty
measured repetitions per configuration. New codecs use seeded random weights.
The inference path uses no LiDAR or labels and asserts zero teacher calls.

Dense Conv2d, Conv3d and Linear hooks count executed MACs; two FLOPs per MAC.
These are lower bounds excluding cost-volume interpolation, normalization,
pooling, activations, KKT allocation and channel arithmetic. GPU stream
elapsed times include host launch gaps. Data loading and file I/O are excluded;
NMS depends on predictions. This is neither an embedded-device energy estimate
nor a real-data deployment latency benchmark.

All three communication configurations transmit 64000 complex data symbols
with total AWGN energy 64000 on this shape. RGB width 40 matches geometry
width 4 through actual tensor dimensions, not equal numeric channel width.
Processed and raw geometry retain every original detector layer and all 72
downsampled depth bins. The raw split moves dres0/dres1 to the receiver;
the posterior head changes input channel count to accept the original raw cost.
The measured reduction therefore is not merely subtracting nominal layer costs.
Both geometry variants reconstruct all downstream appearance inputs from
symbols. RGB reconstructs both images before the complete original detector.

Full forward and task-loss backward checks pass for both new variants, with
strict detector loading and finite codec gradients. They are engineering
checks without optimizer steps. The RGB network is a new controlled comparator,
not Cao et al.'s optical-flow/ROI codec or an exact published DeepJSCC model.

Keep these unfavorable geometry costs in the final system comparison. Boundary
choice requires training-only exploration and later full AP validation.
The optional `StereoStudentEncoder` preserves full-resolution 32-channel stereo
features, quarter-resolution appearance, all 72 depth bins and the complete
original cost refinement/projection/BEV/detection receiver. The original
ResNet feature extractor and neck remain frozen training-only feature teachers.
Both teacher types record zero inference calls. Feature distillation is an extra
training cost and must enter matched-budget comparisons. Full-resolution task
backward reaches all 55 trainable student/codec parameter tensors with finite
gradients; no optimizer steps. Evidence: `full-student-geometry-verification-002`.

A raw concatenation cost volume repeats left features over depth. With a linear
posterior head their contribution cancels in the depth softmax. The corrected
head is Conv3d(64,16,1), GELU, Conv3d(16,1,1), allowing left/right interaction.
An independent constructed-input test verifies that left evidence can change
the posterior. This does not establish calibrated uncertainty. Earlier linear
student measurements (8.88 G Tx lower bound) and their exact source overrides
are retained in provenance; current claims use the corrected 12.48 G result.
Task sensitivity profile is `student-task-compute-profile-001.json` using the
same shape, 3 warm-ups and 20 repetitions. Its predictor uses only pooled
transmitter features. Native-loss gradient supervision runs during training
only, excluding imitation/depth/2D auxiliaries, with detached targets and no
second derivative. Added dense compute is 0.03164 G. Full GPU backward reaches
all 63 new parameter tensors, peak8.421 GB, with no optimizer step. Independent
gradient isolation tests pass. No trained importance or AP claim is supported.
