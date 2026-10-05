# Geometry ambiguity and preventable task damage: pilot001

Prelock before implementation or new measurements. Motivation is the attachment's
geometry+uncertainty+task allocation proposal, qualified by docs/risk-proxy-scope.md.
F6b established usable task-adapted transport; F7 remains a separate matched
channel-training control. Neither validates intrinsic stereo uncertainty as a
communication priority. This pilot tests that necessary premise, without new
optimizer updates, mainval/AP or a claim of optimal allocation/novelty.

Question: does a teacher-free local epipolar ambiguity measure from the clean
transmitter features predict *additional* native3D loss from communication
corruption, beyond feature energy or a shuffled score? Clean ambiguity and
channel damage are distinct. Hypothesis: geometry may explain part of the
incremental damage, but ambiguity alone will not reliably order all groups.
Negative evidence must remain eligible; no post-hoc score/region selection.

## Locked endpoints and sampling

Use sole completeF6b535-state checkpoint77bc73021a139099e3d0f9c12e0fba8b17eb62f9b9c243eb8f0ad099bd6e8867,
existing native student/codec/receiver and62400-symbol layout. Model parameters,
buffers and all module eval flags stay fixed; no teacher,2D/depth loss or optimizer.
Select the first32 IDs in lexicographic order from fixed3340 training IDs for
pilot calibration observations, plus the next32 for an out-of-fit diagnostic.
Both subsets are inside training and disjoint from internal372/main3769.
Selection ignoresGT/scene/AP/entropy; include legal empty-target samples.
Engineering uses the first2 selected IDs and is discarded as calibration fit.
Full64-frame pilot requires an execution addendum committing sampled IDs and
actual native GPU engineering evidence first. Do not queue it on the live
original/F7 GPU scopes. Engineering CPU shape/physics checks are allowed now;
GPU engineering waits current queues terminal and physical margin.

Groups: fixed4vertical×8horizontal cells mapped separately to the three code
layouts; the same cell jointly perturbs both stereo views and appearance.
Every symbol belongs to exactly one group, for32 groups. Mask assignment uses
only public layout, not GT/object masks. No region is omitted. All clean and
intervention passes retain the same sensors/augmented GT/calibration and full
535 readonly state. For this pilot inference uses unaugmented native dataset
inputs; no randomT reuse or GT reaches student/communication/backbone.

At each frame, measure clean native3D loss L0 at the target-only native head.
For each groupj, inject actual independentCN(0,0.1) noise only on its symbols
(Es1/nominal10dB local perturbation), with4 fresh draws and count all62400
transmitted complex symbols per attempt. This is a structured intervention,
not standard full-frame AWGN or a new main-channel result. Source normalization
and codec decoding remain unchanged; no clean-feature receiver bypass.
One extra full-frameAWGN10 draw is a check of observed operation boundaries.
Dedicated PCG64 seed2801 generates the fixed native-layoutI/Q arrays on CPU,
then transfers the actual float32 arrays to GPU; no global/data/model RNG draw.
Retain noise masks/arrays/state SHA. Zero-noise pass consumes no noise draw.

Targets pergroup: signed mean(loss_jr-L0), positive part of that mean, and
sample variance of(loss_jr-L0). Report signed values and4-draw scatter, not
only clipped favorable examples. These are smooth native training-loss
intervention targets, not AP/IoU expectation guarantees or calibrated safety.
Clean squared latent-loss gradients predict first-order variability; report
that separately from mean loss increase. Any later resource test must measure
actual AP and serialized allocation/control costs in its own locked protocol.

## Transmitter-only geometry probe

Use the clean32-channel student stereo features before the existing codec,
stride4 average pooling for a lightweight *diagnostic*, cosine correlation on
horizontal epipolar correspondences; positive disparity hypotheses1..48 in
pooled pixels (=4..192 original feature pixels), temperature0.1 fixed.
Invalid horizontal support is masked before softmax; zero-energy features are
marked invalid rather than assigned confident disparity. This is a chosen
coarse-cost variant, not the receiver's metric-depth sampled cost axis.
No new learned parameters or GT/depth/teacher passed to it.
Public calibration fB maps each bin d to depthZ=fB/d after exact image coordinate
transform. Compute posterior mean/variance ofZ by full discrete pushforward,
not a delta formula at mean disparity. Entropy and disparity variance are
reported as distinct proxies; d0 is not silently clipped into a finite depth.
Left x↔right x-d and right x↔left x+d signs must be independently checked.
Average valid observations inside each fixed cell, preserve valid counts and
empty/invalid groups as undefined with explicit coverage, no forcedzero score.
Receiver never receives these clean transmitter scores in this diagnostic.

CPU checks: known shifted identical synthetic descriptors identify the correct
left/right disparity; true depth moment matches explicit finite-bin expectation;
nontexture/zero features invalid; image-edge support masked; full layouts map
all62400 symbols exactly once; masked/noise RNG replay/global-state isolation;
no GT-shaped input accepted at the geometry probe. Synthetic checks validate
arithmetic only, no learned uncertainty calibration or native task evidence.

## Analysis, after separate native execution addendum

On first32 frames: fit only a ridge linear model for positive damage from log
feature energy, geometry entropy/depth variance/valid support and squared
native latent gradient; gradient is a training-target diagnostic, unavailable
as an inference scorer. Fixed ridge1.0, training-only standardization. Compare
energy-only, geometry-only, energy+geometry, gradient-only, and deterministic
within-frame shuffled geometry seed2802. Identical complete-case support for
allcomparisons; disclose invalid fraction and every signed-damage distribution.
Second32 frames: report heldout prediction error and within-frame Spearman rank
against signed/positive loss increments and squared fluctuation, grouped by
frame (no treating32 correlated cells as independent frames). Bootstrap the heldout32
frames seed2803 with1000 resamples, not symbol-level confidence intervals.
With32 heldout frames and4 noise draws this is exploratory calibration evidence;
no method selection/mainval/AP or multiple-seed communication claim follows.

Do not fit a calibrated inference risk head or allocation policy unless this
pilot's targets/proxies have usable predictive evidence. A future matched-
capacity inference scorer must use sensor features only and charge all extra
supervision/updates plus actual signaling/physical energy. If entropy fails,
revise the scientific hypothesis before spending main-training budgets.

CPU implementation clarification, before any measurement: nontexture means
no local angular variation in the L2-normalized descriptor's four immediate
spatial neighbors; relative descriptor difference squared must exceed1e-12.
Zero-energy descriptor threshold is machine precision (dtype-specific).
Use both valid endpoints for each cosine pair, no invalid-match posterior mass.
Map group cells by floor((index+0.5)*groups/dimension), so normalized cell
centers define all public layouts consistently. Noise draws fullNx2 I/Q
arrays then masks out other groups, making retained RNG exposure explicit.
