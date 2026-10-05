# F5b: native full-resolution stereo feature transport after zero-update interface failure

This corrected exploratory protocol is locked before new execution. The original
F5 protocol remains untouched in stereo-feature-warmup.md. Engineering run
stereo-feature-sanity-001 under implementation5a8356f failed before any update:
its assumed equal quarter-resolution stereo/appearance interfaces were wrong.
Complete source closure and failure manifest are retained. No AP or learning
outcome selected this correction. The actual F2 student in src/geocomm/student.py
returns32×320×1248 stereo and32×80×312 appearance; original native build_cost
uses full-resolution stereo features with receiver-side downsample4. This is a
new explicit layout choice, not a silent change to the old locked architecture. The complete five-condition
pooling suite `pooling-diagnosis-seed17-002` found Moderate3D AP42.3141% for
control,40.3959% appearance-only pooling, .01672% cost-only pooling,0% both,
and .07900% depth-preserving spatial cost pooling. All5×372 native frames,
required519 states, operation metadata, AP and source audits closed before this
protocol. Depth averaging alone does not account for all observed damage.
Do not extend the failed raw-cost representation with allocation comparisons.

## Question and representation

Can fixed-budget transmission of learned per-view stereo features preserve
native matching information better than pooling a preconstructed cost volume?
This tests a communication boundary, not a guarantee or established novelty.

Keep the exact task-adapted F2 student and downstream from checkpointSHA
7e1ceacb4abff0bdb32ef89507b9786e1d80ae9c166ccaaeb4ffb4fcbd639bd2,
including all519 states. Initialize only the new16 codec weight/bias states
freshly withseed17, no F3/F4 codec or engineering weights. Disable the former
raw-cost link; add a link immediately after the two native student forward
calls and before native `build_cost`. Geometry construction and all subsequent
native detection occur at the receiver from received stereo features.
No RGB reconstruction or original-feature/LiDAR teacher at inference.

The shared stereo codec accepts both native32-channel320×1248 feature maps
separately. Preserve all1248 horizontal epipolar locations and use ceil-grid
pooling(16,1), producing20×1248. Conv2d32→32(kernel1), GELU,
Conv2d32→2(kernel1), gives one complex use per grid location and view. The
shared decoder is Conv2d2→32(kernel3,pad1), GELU,
Conv2d32→32(kernel3,pad1), bilinear restore to320×1248,
align_cornersFalse. This allocates fewer channel dimensions and pools vertical
structure; preserving horizontal sampling does not guarantee correspondence
or detection accuracy. It is not equivalent to uncompressed features.

The left appearance stream retains the original F5 choice:32-channel80×312,
ceil-grid pooling(4,2)→20×156, Conv2d32→32(kernel1), GELU,
Conv2d32→8(kernel1); decoder8→32→32(kernel3,pad1), GELU between,
bilinear restore to80×312, align_cornersFalse. No BN/dropout in either codec.
Only16 new weight/bias states remain; total535 and all519 F2 states frozen.

Exact fixed public layout:

    stereo: 2views ×1complex ×20×1248 =49,920complex uses
    appearance: 4complex ×20×156 =12,480complex uses
    total:62,400complex uses per native preprocessed frame.

The public receiver shape description contains both native interfaces, not a
private clean tensor or a sender-derived normalization factor. No upsampled
quarter-resolution input is substituted before encoding. Restore separate
stereo and appearance shapes, then native receiver build_cost follows channel.

Use the established real/imaginary packet convention, joint per-frame unit
average complex energy and uniform power. Receiver receives symbols only;
no clean/private normalization factor, clean feature, power map or true fading
coefficient is passed to a decoder. Public model/layout and static calibration
are the existing preprovisioned assumptions. There are no selection indices
or variable layout in this experiment; header/pilot0 for identity/AWGN. CBR
uses actual6×320×1248 input-real denominator and is1/38.4. A later variable-rate
or dynamic-metadata deployment needs separate counted control and protocol.

## Fixed reconstruction warmup

Exactly1epoch/3340 AdamW updates on the same native original-train3340 split,
seed17, batch1/four spawnworkers, original native LIGA paired augmentation and
empty-augmented-GT resampling retained. GT-dependent preprocessing is inherited
and charged; only RGB/calibration/image/frame keys enter the frozen backbone.
All519 predecessor state values remain bitwise fixed, including native buffers.
Parent/backbone/student/codec eval mode is kept; only16 encoder/decoder weight/
bias tensors require gradients, freshAdamW1e-4, WD1e-4, clip10. Reject nonfinite
loss/gradient before each update. One optimizer step updates the shared stereo
codec using both views; no extra per-view step.

Loss = .5(MSE(left_stereo_received,left_stereo_clean)
             +MSE(right_stereo_received,right_stereo_clean))
       +MSE(appearance_received,appearance_clean).

Targets are detached features of the same frozen F2 student on the same native
augmented stereo input. Channelidentity, uniform62,400uses, training ends
immediately after differentiable feature-link reconstruction capture: no
`build_cost`, native downstream task head or training teacher executes.
Record each actual returned frame, the three distinct native shapes/loss components, power,
complex counts, CBR, LR and preclip gradient. Audit resampling membership and
actual coverage rather than assuming scheduled IDs guarantee returned IDs.
Full535-state checkpoint, all16 actual Adam moments/counters,519 frozen states,
student2/link1/forbidden0 per update and contiguous raw scalars are independently
verified. Whole executable-source identities checked before/after the run.

A separate3-update native GPU sanity (and independent audit) must pass first;
its weights never initialize formal training. Commit implementation before
sanity and sanity evidence before formal launch. Respect the live original
83-epoch cycle's measured GPU memory, leave its isolated sources unchanged.

## Predetermined final evaluation and decision

Use the sole final epoch1 checkpoint, seed17, full ordered372 training-only
heldout frames, sensor-only native inference under (1)identity and (2)AWGN10dB.
Perform complete native prediction/label/calibration/AP_R40_IoU.7 checks and
wire/energy/full-state/source audits for both. Record actual sender/receiver
module boundaries, zero teachers and receiver-only cost construction.
Passive same-input stereo-left/right/appearance distortion statistics accompany
both evaluations; reductions/coverage are independently audited, raw tensors
need not be retained and fresh recomputation must not be claimed.

Report both, charge all prior F1/F2 student exposure and representation-selection
exploration before later matched controls. No best checkpoint, extra epoch,
allocation tuning or main validation. Useful AP and comparable reconstruction
would support a separately budgeted task/noise adaptation experiment; failure
requires inspecting representation/receiver compatibility before allocation.
This warmup itself tests neither calibrated uncertainty, scene/channel-adaptive
rate, risk-theory validity, repeated-seed significance nor method superiority.
Those remain mandatory project work and cannot be replaced by passing this run.
