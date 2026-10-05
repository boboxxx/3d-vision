# Protocol: geometry-dependent latent communication

Status: planned; locked before substantive runs. H1–H3 are confirmatory.
Implementation smoke checks and parameter tuning are exploratory engineering.

## Question and contribution under test

At identical complex channel uses and total transmission energy, does allocation
using posterior depth risk improve stereo 3D detection relative to uniform power
and generic importance? Can direct latent transmission improve the frontier over
the original reconstruction approach while respecting sensor compute cost?

## Full proposed method

Use complete official LIGA-Stereo, preserving its depth, geometry projection,
BEV and detection heads. Place the communication boundary after the initial
cost-volume processing. Encode stereo geometry volume and appearance features
into dense fixed-layout complex symbols. Geometry pooling stride is (8,4,4)
over depth/height/width; appearance stride is 4. A learned codec operates on
these pooled features and reconstructs full tensor shapes for the complete
detector. Pooling is a communication bottleneck, not a change to detector
resolution or depth bins. All downstream appearance features
must come from decoded symbols. No clean skip, GT ROI, exact noise, transmitter
power map, or private latent scale crosses the boundary.

A small transmitter posterior head predicts depth probabilities at the cost
volume's spatial resolution. Train it on sparse projected LiDAR depth using
valid measured pixels; never use ground truth to allocate at inference. Posterior
depth variance supplies a geometric risk coefficient. Calibrate/validate this
posterior; uniform high entropy is not proof of useful uncertainty.

Power allocation minimizes the conditional Gaussian proxy sum a_i/(1+snr*p_i)
subject to nonnegative p_i and sum p_i=N. This is a conditional surrogate, not
a stereo task rate-distortion theorem. Decoder receives noisy symbols directly
and does not undo transmitter power using hidden side information. Changing
latent dimensions changes channel uses; power adaptation alone does not.

This boundary moves some stereo computation to the sensor. Measure this cost
explicitly; the original motivation favors a lightweight transmitter. Revisit
the split if transmitter compute defeats the communication benefit.

Exploratory compute audit on 2026-10-03 found 922.27 GFLOPs transmitter dense
convolution/linear lower bound for this processed-cost split. The optional
`raw_cost` split moves the unchanged dres0/dres1 layers to the receiver and
measures 619.01 GFLOPs. Neither establishes a lightweight sender. Both preserve
the full detector, tensor resolution, depth bins and symbol budget. Boundary
selection is an exploratory decision still requiring training-only tuning and
clean-baseline validation; no confirmatory AP comparison has begun. Keep both
raw measurements, including unfavorable compute evidence. An optional distilled student sender is now implemented as an exploratory
architecture: original full/quarter-resolution feature interfaces, frozen
training-only feature teacher, complete original receiver. Corrected nonlinear
raw posterior permits left/right interaction. Measured student Tx dense lower
bound is 12.48 G; full task backward passes, but it remains untrained and has no
AP or deployment benefit evidence. Training budgets must include its feature
distillation exposures and teacher compute. Optional task-loss sensitivity now passes engineering checks, without
training or AP evidence. It learns per-location positive importance from native
3D classification/box/direction-loss first-order gradients with respect to
received complex symbols, excluding imitation/depth/2D losses. Targets and
auxiliary predictor inputs are detached; no second derivative or inference GT.
`geometry_task` multiplies importance by depth variance before the same proxy
allocation. This combination is an empirical heuristic. Keep identical
sensitivity heads and supervision enabled in uniform/geometry/task ablations
to control capacity and extra supervision; also report no-sensitivity controls.
Match image exposures, optimizer steps and teacher compute before comparisons.

## Data and evaluation

KITTI object stereo: image_2, image_3, calibration, labels and training velodyne.
Use upstream train/val split with SHA-256 and disjointness checks. Keep validation
fixed, use training-only internal tuning, no test-set selection. Preserve the complete upstream receiver and its released 320-pixel image crop. Primary: Car moderate 3D AP_R40 at IoU 0.7;
secondary: Easy/Hard, BEV, Pedestrian/Cyclist, depth MAE by 2–20/20–40/40–59.6 m,
posterior NLL and calibration, latency, peak memory, transmitter FLOPs.
Original-paper AP IoU 0.5 must be reproduced and labeled separately.

## Channel and resource definitions

One complex symbol equals two real scalars. CBR=n_complex/(6*H*W), since input
contains two RGB images. Report n_complex directly too. Unit average complex
symbol energy; AWGN noise variance N0=10^(-SNR/10), N0/2 per real component.
Block Rayleigh uses one fading coefficient per frame. Estimate it from explicit
pilots with noise; count pilot symbols and energy. Full-grid layout needs no
selection indices. Static camera calibration and shared model/rate configuration
are preprovisioned assumptions. Count dynamic headers and calibration whenever
this assumption is relaxed. No claim of physical digital bit rate from analog
latent channel uses.

## Comparisons

1. Clean-link full detector upper reference.
2. Original optical-flow/ROI/RGB method + Stereo-RCNN at original settings.
3. Matched RGB JSCC + the same detector (new controlled comparison, not a
   published DeepJSCC architecture or the original paper's codec reproduction).
4. Same geometry/appearance codecs, uniform power.
5. Same codecs, learned generic importance.
6. Geometry allocation, and geometry times predicted task sensitivity (proposed).
7. Uniform disparity risk versus depth risk; posterior entropy versus variance.
8. Perfect CSI as a labeled ideal reference; pilot-estimated fading as main fading study.
9. Late detection-box transmission with full transmitter compute measured.

All main link comparisons share the detector checkpoint, fine-tuning budget,
train augmentations and resource accounting. Do not treat LIGA as original-paper
reproduction or compare AP under different thresholds as if interchangeable.
The implemented RGB comparator reconstructs both images before the complete
LIGA pipeline, without clean residuals. On 320x1280 inputs its width 40 and
geometry width 4 each use 64000 complex data symbols and energy 64000 in AWGN.
This match follows executed tensor sizes; equal numeric width across different
representations would not match bandwidth. The RGB sender measures 15.37 GFLOPs
dense lower bound. Codecs are currently untrained; these checks establish no AP.

## Training and sweep

First verify the released detector checkpoint on the full validation split.
Then warm up codecs with detector frozen using task loss, then joint fine-tune
with the full detector. Record all trainable/frozen modules. Train across SNR
[-5,20] dB and evaluate [-5,0,5,10,15,20] dB, AWGN and pilot-estimated fading.
Latent widths [2,4,8,16] complex symbols per spatial/depth location; actual CBR
is determined and logged from tensors, not assumed from width.
Seeds [17,29,43], at least 3 independent noise repetitions per condition.
Save predictions for every frame, full evaluator output, checkpoints and raw
link accounting. Use paired frame/noise seeds and paired bootstrap intervals.

## Decision gates

Verify clean baseline before communication comparisons. Investigate any apparent
gain caused by extra training, hidden clean features, oracle CSI/metadata or
different detector. Require consistent geometry-over-uniform gain in the full
validation protocol and depth bins, with uncertainty checks. Negative outcomes
are retained. Do not expand to temporal/diffusion/multi-agent extensions unless
the core stereo communication claim is supported.

## Evidence record

Per run: code revision and diff, dependency/GPU versions, config hash, checkpoint
hash, dataset/split hashes, training schedule, seed, channel model, estimator,
data/pilot/header symbols, total energy, SNR convention, all AP thresholds,
per-frame predictions, timing method, NaN/Inf checks and raw logs.
The released detector range is 2–59.6 m. Any extension beyond this is a separate
retraining experiment and must not be presented as the released checkpoint.
