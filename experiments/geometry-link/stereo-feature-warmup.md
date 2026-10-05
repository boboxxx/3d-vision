# F5: fixed stereo-feature boundary experiment after complete pooling diagnosis

This is exploratory and locked before execution. The complete five-condition
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

The shared stereo codec accepts both32-channel80×312 feature maps separately;
use ceil-grid pooling(2,1), retaining horizontal epipolar sampling, then
Conv2d32→32(kernel1), GELU, Conv2d32→4(kernel1). The same decoder serves both
views: Conv2d4→32(kernel3,pad1), GELU, Conv2d32→32(kernel3,pad1), then bilinear
restore to80×312, align_cornersFalse. The appearance codec accepts the left
32-channel80×312 feature, pooling(4,2), Conv2d32→32(kernel1), GELU,
Conv2d32→8(kernel1); decoder8→32→32(kernel3,pad1), GELU between layers,
bilinear restore with align_cornersFalse. No codec BN/dropout.

Exact public fixed layout, equal stream widths within each view type:

    stereo: 2views ×2complex ×40×312 =49,920complex uses
    appearance: 4complex ×20×156 =12,480complex uses
    total:62,400complex uses per preprocessed native frame.

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
Record each actual returned frame, the three shapes/loss components, power,
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
