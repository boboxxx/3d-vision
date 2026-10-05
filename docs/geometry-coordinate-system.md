# Native geometry coordinates and pooling scope

The current LIGA sender constructs a plane-sweep cost volume whose depth axis
contains **metric-depth samples**, even though some upstream names contain
`disp`. This follows directly from `LigaBackbone.prepare_depth` and the locked
native configuration, not from a new theoretical result.

The shared native range is2–59.6m, `maxdisp=288`, `downsample_disp=4` and
`downsampled_depth_offset=.5`. Therefore the underlying interval is
(59.6−2)/288=.2m and the72 raw-cost sample centers are

    Z_i = 2 + (i+.5)*4*.2 = 2.4+.8i metres, i=0,…,71.

These centers are converted to image disparity using the calibration-dependent
fB/Z before constructing the cost volume. They are not uniform disparity bins.
Cost input shape is[1,64,72,80,312]. Geometry pooling(8,4,4) produces9×20×78
sites. Because72 divides by9, each depth pool averages8 adjacent raw-cost
samples: center-to-center span5.6m, bin width6.4m. This is a description of an
operation on learned features, **not a claim that predicted boxes have6.4m
error**, a quantization bound, or a posterior probability distribution.

The proposed disparity perturbation relation ΔZ≈−fBΔd/d² remains a local
relation for perturbations of disparity. It cannot be substituted directly
for the effects of pooling the native depth-sampled64-channel cost feature.
Depth-preserving spatial pooling(1,4,4) is consequently a useful diagnosis,
but it is not a rate-matched communication method: at unchanged latent width
it would need8times as many geometry sites. Any later alternative must account
for its actual symbol count, position/mask information and trained receiver
behavior before claiming a rate–geometry benefit.

For example, the current geometry code uses4 complex symbols per site:
9×20×78×4=56,160, plus appearance20×78×4=6,240, total62,400. Preserving all72
depth samples on20×78 spatial sites instead gives449,280 geometry symbols,
455,520 total. Trading spatial resolution for depth at5×39 sites would restore
the same56,160 geometry count, but that changes spatial information strongly
and has not been trained/evaluated. Neither arithmetic establishes a sufficient
representation or a useful allocation algorithm.

Receiver camera calibration and shared model/rate definitions are declared
preprovisioned in `experiments/geometry-link/protocol.md`. Per-frame changing
calibration, crop/shape metadata, masks or layout must be sent and counted if
that deployment assumption is relaxed. Original and proposed methods must use
the same deployment assumptions in a physical-budget comparison.

Sources: local native implementation and locked configuration,
`third_party/LIGA-Stereo/liga/models/backbones_3d_stereo/liga_backbone.py`,
`third_party/LIGA-Stereo/configs/stereo/dataset_configs/kitti_dataset_fused.yaml`,
`configs/diagnostic/student_uncompressed_holdout.yaml`. All executable sources
remain frozen while the five-condition pooling suite runs.
