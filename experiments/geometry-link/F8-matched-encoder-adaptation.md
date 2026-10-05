# F8: matched codec-only versus joint encoder adaptation

Prelock 2026-10-04 after closed F7 and full64 risk results, while the fresh8
paired-noise audit is pending. The choice follows the direction review and does
not depend on fresh8 correlations. Hypothesis: allowing the existing lightweight
student to adapt jointly with the codec improves noisy-channel native 3D AP
beyond the same number of codec-only updates. This strengthens the uniform
baseline; neither extra optimization nor unfreezing is a novelty claim.

Both arms start from the complete F7 AWGN final checkpoint, without optimizer
reuse: `/mnt/d/paper6/runs/stereo-channel-native-seed17-001-awgn/checkpoint_epoch_1.pth`,
SHA256 `ea93fb1ec7d7328e0ae603ec40385882da255414e876efa2c1abda4fa4de36a3`.
Require the sealed F7 training, paired-data, four AP and local artifact closure.
This explicit new initialization does not change the F6b parent of either risk
diagnostic. No engineering weights initialize formal training.

Control updates the same 16 codec tensors as F7; joint treatment also updates
the existing 35 student encoder tensors. No new parameters or architecture.
The full model has 535 state tensors; control keeps 519 and joint keeps all
484 original detector states exactly fixed, including buffers. Both remain in
eval mode, with autograd enabled for the selected parameters only. Verify exact
parameter names, shapes and membership before execution. Shared student runs
on both images; gradients must reach every selected tensor, while inactive
parameters retain no gradients. Report finite/nonzero gradient counts and all
actual Adam states rather than inferring optimization from a loss curve.

Each arm gets one native epoch: 3,340 actual updates on the identical fixed
training fold, batch1/four spawn workers, model/data seed17. Fresh AdamW per
arm: LR1e-4, weight decay1e-4, global gradient clip10 across the arm's selected
parameters. Different clipping norms are part of the declared optimization
scope treatment; record them. Identical 3D classification/box/direction/IoU
objective, separate GT introduced only at the head; retain legal empty-GT
background-only loss. No teacher, depth, 2D, imitation or reconstruction loss.

Both train with uniform AWGN and the same dedicated Python Random(1717)
U[0,20]dB schedule, length3340. Independent CUDA Generator(1718) supplies noise;
reset it to the same initial state for each arm. These seeds differ from F7.
Do not touch model/data RNG when drawing channel noise. Verify per-step actual
augmented left/right images, targets, public calibration, image_shape and
random_T hashes and noise RNG progression across both arms. Seed17 reuses the
earlier deterministic data ordering/augmentation setup; disclose this shared
training history. No metadata fingerprints enter model inputs.

Keep the native stereo/app layout and all 62,400 complex uses per frame
(49,920 stereo/12,480 appearance), normalized joint average energy1. Uniform
channel, no header/pilot in this AWGN comparison, no allocation. Receiver builds
cost solely from received stereo features and uses received appearance. Both
use the same inference graph and parameter count; report measured training
memory/time because the joint backward pass costs more computation. Equal
updates/data/resources do not imply equal training FLOPs.

Evaluate only each final checkpoint on all ordered372 internal hold-out frames,
under identity and AWGN10, evaluation seed17, unchanged native inference and
KITTI evaluator: four endpoints. Primary quantity is joint minus codec-only
Car Moderate3D AP_R40 at IoU0.7 under AWGN10, in percentage points. Report full
Easy/Moderate/Hard, both identity endpoints, errors and failures. No best epoch,
early stopping, SNR/seed search, additional epochs or mainval-based tuning.
A positive point difference is exploratory single-seed internal support only.
Internal author-pretraining overlap remains; no final generalization claim.

Before formal training, run six native engineering updates for each scope from
the same parent, including the seed17 empty-GT case. Charge all12 updates as
engineering exposure and discard their learned weights. Require complete535
parent load, exact frozen subsets, six actual Adam steps, paired actual inputs
and noise, target/receiver boundaries and finite complete selected gradients.
Meaningful CPU checks must exercise scope rejection, joint student gradients,
noise isolation and preserved control behavior. Preserve any failed attempt;
repair interface errors under a separate prelock before a new unique run.

Run sequentially on sheng after fresh8 native/audit/statistical closure. Require
physical free GPU memory >=12GiB and peakreserved+2GiB <= initial physical free;
never interrupt another workload. Use unique outputs under `/mnt/d/paper6`.
Source identities cover root executable/config trees, native dependencies,
F7 reused helpers and new F8 code. Freeze these from launch through all four
endpoint audits. Original21 sources and old protocols remain unchanged. The
closed risk code remains immutable. Implementation may subclass the observer
to permit joint student autograd, but must preserve GT and received-only checks.

Independent closure must verify full model states, actual Adam counts/moments,
all3340 raw training rows per arm, paired actual inputs/noise schedules, native
AP predictions/GT/calibration, source identity and actual terminal processes.
Record results and update the direction review whether positive or negative.
Geometry-dependent allocation still requires its own matched-resource controls,
main validation, fading and repeated seeds after this baseline comparison.
