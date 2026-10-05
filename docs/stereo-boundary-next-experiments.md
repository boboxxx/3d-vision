# Research gates after native stereo-feature transport

This is a design note, not a prelocked training protocol or a completed experiment.
The F5b fixed 62,400-symbol layout compresses each full-resolution stereo view
32→2 real code channels on a20×1248 grid, and appearance32→8 on20×156. It
preserves native horizontal locations but pools vertical structure; decoder
restoration does not recover discarded information by definition. Useful AP,
calibration and communication advantage remain empirical requirements.

## Separate task compatibility from wireless adaptation

Before allocation studies, test whether the existing fixed representation can
be trained to supply the unchanged receiver's 3D objective. Use an explicit
new prelocked budget and the sole audited F5b final checkpoint. An independent
native sanity must check actual full-model gradients and GPU memory. Do not
reuse F3's broad native training objective or its string-based parameter filter:
new names contain `stereo_feature_link`, while the old filter selects
`semantic_link`, which would train a different scope.

A candidate clean compatibility stage trains only the existing16 codec
parameters. Keep all519 F2 states—including global_step and imitation
normalization—fixed, all modules in eval mode, and native paired augmentation
with its geometry transformation. Run the native backbone, height compression,
BEV backbone and3D head explicitly, with gradients enabled. Supply augmented
GT boxes only to the3D head after sender/channel/receiver feature computation.
The native head can assign targets in eval when GT boxes are present; its
imitation branch requires training mode and therefore does not execute.
Native `get_loss` then contains classification, box and direction losses,
without depth/2D/image reconstruction/LiDAR imitation or teacher calls. This
needs native execution, counter, scalar, gradient and exact-state verification;
reading the code alone does not establish successful training.

Keep clean task compatibility and subsequent noisy-channel adaptation as
separate committed experiments. Their budgets, initialization exposure,
losses and noise distributions must be charged to every later matched control.
Compare predetermined final identity and AWGN outputs for each, without
choosing a checkpoint or validation noise seed after observing results.
A fixed receiver failure can motivate a separately controlled receiver-adaptation
study; it does not establish a channel-noise explanation or a fundamental rate
limit. Do not compare an adapted method with an unadapted frozen baseline.

## Risk calibration and resource controls

The previous raw-cost risk estimator is not transferable evidence for the new
sender boundary. A sender may use a small stereo-matching/posterior branch to
estimate depth ambiguity from clean RGB/student features; receiver cost volume
is unavailable to that sender. Profile the actual complete sender, including
this branch and any per-frame allocation computation. A small learned feature
adaptor or cosine matching is not by itself a novelty claim; see the primary
GraftNet note and the existing literature audit.

Depth ambiguity, channel-induced feature error and preventable 3D task damage
must be measured separately. Fit uncertainty and damage supervision only on
the fixed training/calibration folds. Use independent channel perturbations
and an explicit counterfactual transmission intervention to define preventable
damage. Record how perturbations are coupled across conditions; the same
random seed alone does not guarantee the same complex-symbol noise under a
changed layout. Depth variance times squared native loss gradient is a
heuristic, not an expected-loss or AP theorem; see risk-proxy-scope.md.

Required comparisons include uniform, depth-only, task-only, joint risk,
matched-capacity generic scoring and a shuffled-risk control. Charge predictor
capacity, privileged training supervision, extra exposure, symbols, pilots,
control metadata and energy equally. Predictive correlation alone is
insufficient: the calibrated rule must improve actual task outcomes under the
same physical resource constraint, with predetermined multi-seed evaluation.

Fixed full-grid power modulation can avoid an index header only when the
receiver truly uses received symbols and public fixed policy/layout, without
clean/private per-location power coefficients. Its actual decoder error law
must be measured; it is not automatically scalar MMSE or known water filling.
Dynamic selection, rate/layout and channel feedback require counted control
and a new receiver interface. Coherence assumptions, pilot-based channel
estimation and feedback costs must be explicit for fading/adaptive cases.

The original full-native staged9-channel baseline currently uses a different
physical duration from the new62,400-symbol layout. Completing it is necessary,
but its unmatched point cannot establish a same-bandwidth advantage. Establish
trained comparable operating points and report actual per-frame/average uses,
energy and adaptation exposure before any superiority claim. Main validation
is reserved for the locked comparisons; exploratory heldout results retain the
author-pretraining limitation.
