# SRCNN KITTI adaptation and three source operating points: prelock

Lock before KITTI SRCNN training, reconstructed-cache inspection or AP.
This remains a **declared SRCNN adaptation** of Cao's unspecified compressor,
with the already checked official architecture/weight layout. The benchmark
requires more than running the demo x3 model at three arbitrary image sizes.
No exact published comparator, matched-exposure gain or novel method claim.

## Model, data and fixed training

Use the public upstream 3712 training IDs only, exact split SHA256
b6417a1d9b18c8fdb085128e633d28ff321b7674a6d1b3841b8f43d865b281cb.
Each left/right native RGB8 view is a separate training item:7424 items.
No validation image, label, calibration, LiDAR, ROI or downstream detector
is a training input. Disclose the previously inspected clean main results
and F9 internal outcomes; neither is used to select these rate parameters.

Three separate models, nominal10/30/50, each initialized from the identical
official9-5-5/ImageNet/x3 MAT file. This is the released demo default chosen
before the interface checks, not a performance-selected checkpoint. Interpret
the six arrays using the closed author core, then cast once to float32 and
register exactly six trainable parameters. Kernel9/5/5, channels1/64/32/1,
replicate boundaries, two ReLUs and final linear Y. No extra residual path,
normalization, learned resizing, chroma model, attention or task supervision.

The source and receiver preprocessing is exactly the closed declared interface:
floor(H/sqrt(C))×floor(W/sqrt(C)) RGB8 PillowBICUBIC, then float mode-F channel
BICUBIC reconstruction at native H/W, explicit studio-range float64 YCbCr.
Cast reconstructed Y to float32 for training. Target Y is the clean training
RGB8 view converted by the same matrix before any downsampling. Neither clean
color nor image shape outside the received header enters formal reception.

Twenty complete epochs per model; each epoch a dedicated NumPy PCG64 seed17
shuffle of all7424 items, continuing one generator stream. Batch of4distinct
views, exactly8patches per view, total32patches per optimizer step. Each patch
is49×49, with independently uniform legal integer top/left coordinates from
a separate PCG64 seed1719 stream. Use original orientation, no resizing or
flipping after the full-view source preprocessing. Every item is visited
exactly once per epoch;7424 is divisible by4. Exactly1856steps/epoch and
37120updates/model,111360updates total. Replay the same view order and patch
coordinate streams across all three operating points.

Loss is mean squared error between the central33×33 predicted Y and target Y.
The discarded8-pixel patch boundary equals the9/5/5 receptive-field radius,
so patch replicate boundaries cannot affect retained target pixels. Full-image
reception later retains all native pixels, with the author replicate boundary.
Target/evaluation crop is not silently applied to detector coordinates.

Fresh Adam per model, learning rate1e-4, betas(.9,.999), eps1e-8, weight decay0,
no learning-rate schedule, gradient clipping or mixed precision. Fix
torch seed17 and CUDA seed17, batch workers0, CPU threads2. Every one of the
six gradients must exist and be finite; zero gradients are legal and recorded.
Keep the same preprocessed input/target/coordinate identity at matched steps
across rates, except for the explicitly rate-dependent degraded Y. Record all
training losses and six parameter/gradient/optimizer counts, actual view IDs,
patch coordinates and input/target hashes. Author MAT and previous source trees
remain unchanged. No early stopping, best-epoch selection or retry draws.

Only the final20th-epoch weight is eligible for reception; save full six-state
weight and Adam state, exact update counts/source identity. Intermediate logs
are evidence, never alternate selected models. Nonfinite state aborts the run,
retaining its failure ID; any repair needs a new narrow prelock/execution ID.

## Engineering, compute and result release

Implementation lives outside frozen project/src and legacy detector roots.
CPU tests first check batched float32 architecture against independently cast
author filters, patch-central/full-image agreement, gradient scope, training
sampling and byte-only reconstruction using the actual source interface.
Discard engineering weights. Native engineering is exactly6updates/rate using
only training IDs000000/000003 and both views, same sampler coordinates across
rates, complete optimizer/weight/source audit. Full formal training cannot
start until this gate closes and existing source-cache main detector cycle is
actually terminal. Do not interrupt that cycle or run a competing GPU trainer.
Require physical free>=12GiB and measured peak reserve+2GiB<=initial free.
Artemis training requires a separately verified native GPU runtime first.

Full three-rate formal training and independent all-update/optimizer/terminal
audit precede any main reconstructed SRCNN cache or detector result. Native
raw received RGB cache format is float32 HWC, converted once from the explicit
float64 inverse color operation. Preserve negative/above-one values, report
range and conversion identity. This differs from the JPEG/JP2 RGB8 cache and
requires a separate float-cache adapter and native engineering gate.

Main source-only evaluation is all3769 public validation pairs per rate.
Generate actual wire files first; new receiver processes have only received
bytes, static final model and source-independent implementation. Record32-byte
headers, all payload bytes and actual ratios; no radio uses/energy assumed.
Independent pixel/parameter/geometry audit then both author downstream models,
no GT until complete prediction sets. Their decoder and native model parameters
stay fixed. Native-frame/source-quality/ROI policies remain separate addenda;
never substitute clipped uint8 for the declared float task input.
Report all three rates and E/M/H results, including failures. Original right2D
ambiguity, other sources, full wireless matrix, matched new-method comparison,
multiple seeds and manuscript remain part of the active project objective.
