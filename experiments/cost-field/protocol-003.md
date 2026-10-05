# Native voxel coordinates, bounded receive decoding and a stronger generic control

Prelock before code003 CPU/native execution or training. Preserve GPU001 and
GPU002, including GPU002 job11424908 FAILED1:0 at the historical no-cut
prediction tolerance on the second source frame. This failure does not measure
the corrected kernel: no corrected codec case ran. Do not relax its threshold
or present that attempt as passed. The previously closed GPU001 cut/backward
gate remains evidence for the frozen placement, not corrected-code accuracy.

## Coordinate contract

The unchanged native voxel sampler normalizes camera depth by CV_DEPTH_MIN/MAX
and calls grid_sample(..., align_corners=True) on a D=72 feature tensor.
Its effective feature-plane query vertices are linspace(2,59.6,72), whereas
the raw cost sweep has centers2.4..59.2 and the upsampled depth head centers
2.1..59.5. Declare all three separately. Attach conditional feature atoms to
the effective native voxel-query axis; encode means and evaluate receiver
kernels on this same public axis. The low-resolution native softmax guides
feature selection; it is neither calibrated depth nor an exact distribution
of the upsampled depth estimate. Do not change author tensors or sampler.

## Receive-domain constraint

Keep the raw AWGN/Rayleigh arrays and perfect-CSI untruncated ZF unchanged.
Project received feature groups to their known transmitter spheres before
decoding: energy4 for each node and appearance, energy1 for position pairs.
This uses only received vectors and public energies; source norms/gains do not
cross the channel. Use scale-first projection to avoid finite-large input
overflow, and a fixed zero-vector fallback. It bounds decoder input under
deep fades without clipping fading/channel noise or claiming finite ZF MSE.
Use the same receive projection in every arm. Keep exp-kernel/K from002.

## Generic whole-codec baseline B

G remains the prior ablation; it is not the primary generic control. B receives
the same pooled32-channel cost, original pooled native probability and pooled
appearance as P. Concatenate centered log-probability as channel33, then
Conv3d33->10, SiLU, learnedLinear72->3. Transmit30 real values as15 complex
symbols/site on an energy15 sphere. Decode with learnedLinear3->72, SiLU,
Conv3d10->32, followed by the same spatial interpolation. Appearance uses the
same32->8->32 architecture and four energy4 complex symbols. All four arms
therefore send19 complex symbols/energy19 per site (29640 per real frame).

B has1751 functional parameters vs P/G/S1650, a deliberately stronger control
with greater depth degrees of freedom. Do not claim equal parameter counts;
do not pad parameters. Its native-prior input prevents attributing P's benefit
merely to access to a pretrained predictor. P-B is the eventual primary
geometry contrast; P-S controls spatial organization, P-G is secondary.

## Engineering optimizer gate, not a confirmatory AP experiment

After CPU checks, run eight actual Adam updates per G/P/S/B, lr1e-3, no weight
decay, gradient norm clip10, float32, original484 states frozen/eval. Alternate
the same training frames000000/000003 in the same order for every arm.
Conditions in order: identity, identity, AWGN6, AWGN10, AWGN18, Rayleigh6,
Rayleigh10, Rayleigh18. Common noise seed800001+step. No validation/AP selection.
Initialize G/P/S from the same retained001 codec weights, replace only the
public depth-axis buffer. B uses seed17 and common appearance weights.
Train only the new codec, using real native cls+box/IoU/direction loss.

Sensor-only forward keys and received-only boundary remain unchanged; labels
are read loss-side after forward. Record every raw wire, gradient, optimizer
state/checkpoint, target and loss; no partial sampling in closure. Require
finite loss/gradients, finite raw received arrays, changed new codec weights,
unchanged author484/source/runtime, actual terminal, and full transferred
record proof. Do not require lower loss on a different frame/channel or infer
generalization from these32 updates. Save failure records before assertions.

Historical no-cut predictions are recorded diagnostically when available;
there is no new declaration of their numerical equivalence and no change to
the failed002 criterion. Full train3712, val3769, multi-seed geometry effects,
matched resource curves and original complete wireless comparisons remain
subsequent work. This gate starts actual optimization rather than promoting
cold outputs or synthetic tests to research results.
