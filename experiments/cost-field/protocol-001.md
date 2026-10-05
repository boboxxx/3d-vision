# Conditional metric-depth cost-field communication: locked mechanism prototype

Previous matchedF9 and probability-onlyQ1/Q2 results do not establish a
geometry-specific benefit. Full native author detector/task-loss access on
Artemis is now verified. Test a new representation intervention, preserving
original stereo sender→wireless→cloud3D task scope and the complete original
main validation/rates/AWGN/Rayleigh matrix as eventual requirements.

The sender computes the author's final32-channel72-depth cost field and left
appearance from both images. The communication cut is after the final stereo
hourglass, before native depth prediction, metric voxel sampling and detection.
After the cut, every depth probability/3D feature must derive from received
cost/appearance. No clean cost, appearance, posterior or GT bypass is allowed.
Original calibration and tensor/grid shape are the common public metadata;
new dynamic nodes/appearance travel through the charged noisy payload.

At a common4x4 spatial pool, represent each ray byK=3 equal-mass conditional
depth barycenters and eight learned feature coordinates per node. For source
p_j over known metric axisz_j, m_kj is the intersection length of CDF atomj
with[k/K,(k+1)/K]. Send mu_k=K sum_j m_kj z_j and node value
v_k=K sum_j m_kj encoder(C_j). This uses continuous interval barycenters,
not discontinuous point quantiles or the unsupported probability-onlyQ2 code.
It does not claim cost features are a calibrated probability or sufficient3D
statistic. Equal-mass W2 quantization is a mathematical ingredient, not novelty.

All arms share identical trainable residual score head, feature projection/
decoding, appearance projection/decoding and metric RBF decoder. P adds the
sender's native low-resolution depth-head probability to the learned scores;
G uses only the learned generic scores; S adds spatially rolled native prior
(half-grid roll in both spatial axes). All compute the native source head for
matched exposure/compute. Each receiver uses only corrupted node/feature symbols.
The original full detector states stay fixed; no sparse teacher/GT enters it.

One complex unit-energy phase symbol communicates each metric node with an
arc[-pi/2,pi/2]. Each node's eight feature coordinates occupy four complex
symbols normalized to total energy4; appearance uses four further symbols
with energy4. Every pooled site therefore pays exactly19 complex uses/energy19,
including node coordinates. Zero-feature normalization uses a fixed unit-vector
fallback. No clean gain/norm/map/index is sent outside the payload. This is an
explicit analog engineering choice; prior geometric analog coding prevents
calling phase/sphere normalization new. G gets the same coordinate slots and
can learn depth organization; fixed empty-coordinate padding is not its control.

AWGN complex noise has E|n|^2=10^(-SNR/10); original main Rayleigh is iid complex
CN(0,1), perfect CSI and untruncated zero forcing. Pilot-estimated fading is a
separate practical extension. Shared physical draws across arms are mandatory.
Received coordinates are bounded to the public depth range; no sender-private
probability or noiseless validity mask helps decode them. Receiver RBF bandwidth
is shared trainable scalar, initialized from the metric span/K. Features are
interpolated to original native dimensions for the unchanged receiver.

Immediate gate: implement meaningful CPU tests for CDF barycentric continuity,
exact symbol/energy accounting, received-only independence and gradients; then
run the actual native cut and detection loss/backward for all three arms on the
same two fixed training frames000000/000003, identity andAWGN10. This is an
engineering gate, not a training/AP/novelty result. Save all transmitted/received
symbols and reconstruction descriptors, losses/new gradients/states; retain
failed attempts. No post-hoc best-arm selection or detector-state adaptation.
Native identity callback must reproduce the original full detector to numerical
tolerances already observed; a false bitwise identity claim is excluded.

Only after the cut is real and usable, lock matched source-feature warmup and
actual3D-task training, train-only choice rules, multiple seeds and full main
evaluation. Primary science will compareP−G at matched uses/energy/exposure,
andP−S for geometric organization. Extra capacity, learned generic pooling or
original task pretraining cannot alone support the contribution. Preserve
negative outcomes; do not abandon the full original experiment scope.
