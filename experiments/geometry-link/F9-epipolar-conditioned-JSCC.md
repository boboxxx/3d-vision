# F9: correspondence-coupled coding, directly tested through 3D AP

Prelock before F9 implementation or measurements. F8 is fully closed with
matched AWGN10 Moderate+3.2864points for joint adaptation. Old coarse geometry
scores lack supported mean-damage prediction. Test a different operational
hypothesis: coupling how corresponding stereo regions are encoded improves
task performance beyond an equally parameterized generic coding head. Do not
fit or use the old mean-risk/variance targets as priority supervision.

This is geometry-conditioned JSCC. Learned gains alter the coded content as
well as its energy; the receiver does not invert them. It is not pure unequal
error protection, dynamic symbol rate, calibrated uncertainty, or an already
novel contribution. Main-validation/original-scenario evidence remains required.

## Fixed model and four arms

All arms start from the same complete535-state F8 joint checkpoint
6e54ebc5fa696db26b3902cdd129ff42544635db4cfb56dfdb7f4b94841c5193,
/mnt/d/paper6/runs/stereo-encoder-native-seed17-001-joint/checkpoint_epoch_1.pth.
Never select/restart from an intermediate checkpoint. Preserve old source trees;
new code and any config live under experiments/geometry-link/F9 only.

U: uniform coding, continued student+codec training. G: generic shared left/right
gain head, each view independent. P: identical head plus true epipolar coupling.
S: identical head plus column-shuffled correspondence coupling. G/P/S have exactly
the same new parameters. U has the same newly initialized state tensors but keeps
the head frozen and bypasses its effect; capacity differences are controlled by
G, so P>U alone cannot establish a geometric contribution. Report actual extra
sender compute/memory/time; do not claim equal training or inference FLOPs.

Keep the existing16codec and35student tensors active in all arms; all484original
detector tensors/BN states fixed, eval mode throughout. New shared gain head is
Linear33→16, GELU, Linear16→1: four parameter tensors,561scalar parameters.
Input is the32-channel mean feature in each of32public4x8spatial groups plus
nominal SNR/20. Group membership uses pixel-center coordinates; average actual
member pixels, do not use adaptive pooling's overlapping boundary bins. Share
the head across views; appearance has no gain head. First linear uses standard
PyTorch initialization inside an isolated CPU RNG scope seed1930; last weight
and bias are zero. All four arms initialize these tensors identically without
advancing global training RNG. Parent535states load strictly before registering
the head; verify all535 equal. Full state count539. U trains51/fixes488;
G/P/S train55/fix484. There are no extra learnable temperatures/mixing weights.

## Correspondence computation and coding

For P/S, average clean sender stereo features over nonoverlapping4x4image cells.
Require batch1,32channels, equal H/W divisible by4. Normalize each32-vector with
L2 denominator clamped at1e-6. Compute cosine scores along the same row at input
disparities0,4,...,192pixels (pooled shifts0..48), masking only out-of-canvas
correspondences. No GT, teacher feature, calibration-derived depth label, object
ROI or content-dependent validity mask. Public padded canvas participates in the
representation; do not claim all positions are real-image support. Zero/flat
features have defined ambiguous scores. d0 ensures each row has valid support.
Temperature0.1, masked softmax separately for each view. Use opposite shift sign
for right-reference candidates. Gradients pass through this entire calculation.

Aggregate probabilities into32x32 row-stochastic matrices M_LR/M_RL: map source
and target pooled-cell centers to the same public4x8group grid, sum probability
mass, average across reference cells in each group. Check every group nonempty
and row sum1 within FP32 arithmetic. All matches remain within the same vertical
group. For a group logit q, P uses
bL=(qL+M_LR qR)/2 and bR=(qR+M_RL qL)/2.
G uses bL=qL,bR=qR. S uses the same matrices but rotates target columns by4
within each eight-column vertical group before multiplication (fixed deterministic
permutation, preserves row mass/entropy, breaks the actual correspondence).
No sample-dependent permutation selection or extra RNG. This shuffle is a
counterfactual coding control, not a valid stereo geometry model.

Amplitude gain per group is exp(log(2)*tanh(b)), in[0.5,2]; U uses1. Map gains
to the existing stereo encoded grid by public cell-center group membership.
Apply the same amplitude to each I/Q pair. Appearance gain is1 before common
normalization. Pack the unchanged code layout and normalize the actual complete
symbol vector to unit mean complex energy exactly as the existing link does.
All arms attempt62400complex uses:49920stereo+12480appearance, zero headers or
pilots for these identity/AWGN conditions. Record actual64stereo-group energies,
appearance energy and total energy; gains themselves are not measured powers.

Receiver input remains only received symbols and public fixed feature layout.
No q, M, gain, normalization factor, clean feature or extra side channel reaches
decode(). The existing learned decoder handles the changed representation.
Nominal SNR is public task configuration, not instantaneous fading CSI. At
identity evaluation use nominal SNR10 so its sender condition matches AWGN10.

## Training, inference and falsification

Four arms each3340updates over the same existing internal training split,
batch1/workers4spawn, native model/data seed17. Fresh AdamW1e-4,weightdecay1e-4,
global gradient clip10 over each arm's selected parameters. Native3D
classification/box/direction/IoU loss only, all62legalemptyGT retained. No RGB,
depth, teacher, geometry-proxy or AP-surrogate loss. Training SNR is dedicated
Python Random1927 U[0,20], channel noise isolated CUDA generator1928, same actual
inputs/augmentations/calibration/GT and actual noise per step across all arms.
Selected gradients must exist/finite; initial hidden-head gradient zeros from
the zero final layer are valid and recorded. Full Adam count/moment and frozen
state audits are required; report zeros, do not erase samples or retry draws.

Final-only evaluation: each arm on372internal holdout frames under identity,
AWGN6,AWGN10,AWGN18, fixed inference seed17. Exactly16endpoints; complete all
four arms before inspecting/reporting the comparison. Primary P−G Car Moderate
3D AP_R40 at IoU0.7/AWGN10. Required controls P−S and P−U, all E/M/H and all
channels, actual symbols/energy, feature validity and frozen-state audit. Record
identity changes to distinguish content coding effects from noise robustness;
an identity-only improvement cannot establish stronger channel protection.
No mid-run changes to temperature, grouping, gain range, mixing strength,
training length or checkpoint. Do not select the best arm/checkpoint by internal
AP and silently relabel it as the prelocked proposal.

If P does not improve on G or cannot beat the shuffled correspondence control,
the current coupling hypothesis lacks support. Preserve that outcome instead
of adding proxy fits or retuning the same experiment. Positive internal results
remain exploratory single-seed evidence with author-pretraining overlap.
Main3769validation and multiple seeds require a subsequent prelock before those
results; keep all controls and disclose any promotion decision. Soft-vs-hard
correspondence and varying-vs-fixed SNR-input ablations are necessary before
claiming uncertainty or scene-channel conditioning contributions. Original
AWGN6–18 and per-symbol Rayleigh/perfectCSI/ZF matrix remains separate required
work; do not substitute the existing frame-Rayleigh/noisy-pilot channel.

## Engineering and audit gates

Pure CPU tests first: independent dense correspondence reference for both signs,
known shift, borders/flat features, group normalization/shuffle, exact U and
zero-head parent behavior, actual62400/energy accounting, isolated RNG,
receiver no-side-information barrier, every intended gradient and scope.
Native engineering is fixed six updates per arm from the same F8 parent,
discard all engineering weights. Verify complete input/noise pairing, all539
states, actual optimizer scopes/counts, raw group-energy records, no forbidden
teacher/downstream input, unchanged old source identities, physical free GPU
memory>=12GiB and peak reserve+2GiB<=initial free. Capture one passive sender
compute/timing profile per arm with identical shape/settings; timings are local
engineering measurements, not a deployment latency guarantee.

Only after all native engineering gates close, launch one formal four-arm cycle
with unique output paths and frozen sources. Independent full raw training,
checkpoint/Adam, receiver chronology, prediction/GT/calibration/AP and terminal
process audits precede claims. Preserve failed attempts, error traces and any
separate prelocked repairs; never relax correctness criteria after measurement.
