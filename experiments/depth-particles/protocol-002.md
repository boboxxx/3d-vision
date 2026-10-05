# Q1: mode mass at the same two-symbol budget

Exploratory representation experiment locked before execution. Q0 showed that
three equal-weight quantiles lose the probability of distant modes. Compare
two ordered metric-depth locations a,b and explicit mass m to Q0 using the
same three real source coordinates, four-real charged carrier, two complex
symbols/energy2, received-only decoder, random noise/fades and CPU budget.
This is quantization/communication feasibility, not a claim that optimal scalar
quantization or sphere coding is novel or that this replaces LIGA features.

For a piecewise-uniform histogram with quantile Q, a fixed split m has optimal
L1 atoms a=Q(m/2), b=Q((1+m)/2). Minimize its exact W1 over m in[0,1]. Within
regions where Q(m),Q(m/2),Q((1+m)/2) keep their bins, the objective derivative
is 2Q(m)-a-b, an affine function. Enumerate boundaries from CDF_j,2CDF_j,
2CDF_j-1, and interior derivative zeros; evaluate the exact objective at every
candidate. This gives the global minimum of this piecewise quadratic objective,
including one-sided changes across zero-density gaps. Ties within1e-11m choose
the smallest m deterministically. Derivation and independent verification are
required; do not treat search code as mathematical proof.

Normalize a,b with the same public bounds and m to2m-1. The domain is two
ordered bounded depth coordinates and an independent bounded mass coordinate.
Transmit sqrt2*(q1,q2,q3,1)/sqrt(1+||q||²). Decode the received ratio with the
same1/sqrt2 anchor floor; Euclidean-project the first two coordinates onto the
ordered interval, and clip the mass independently. Do not sort mass with depths.
Pointwise normalized-coordinate error stays<=2sqrt2||effective noise||. A
coupling gives W1 between the two-node laws<=span/2*(||depth_coordinate_error||₂
+|mass_coordinate_error|)<=2*span*||effective noise||. Full histogram error also
contains intrinsic approximation; no detection or finite Rayleigh-noise moment
guarantee follows. Include every actual charged symbol and energy.

Use all twelve previous fixed72-bin sources and one additional three-mode
source, equal masses at bins7,37,67. Keep original Q0 untouched. Primary
feasibility question: does mode-mass transmission reduce the two-mode50/50 and
minority-mode intrinsic error at the same budget, and what does it sacrifice
on uniform/broad/three-mode sources? Report every source and condition, not only
the motivating case. Identity/AWGN/Rayleigh6/10/18,64 common PCG64seed17 draws,
perfect received CSI/unclipped zero forcing exactly as Q0. Save both methods'
complete physical/decoded arrays and intrinsic/channel-only/full W1 separately.

Check global approximation against an independent sorted240000-point inverse-
CDF sample: prefix sums enumerate every possible two-cluster median partition
in O(N). The histogram-to-midpoint sample W1 is at most span/(2N), so optimal
objective disagreement must fit that bound plus rounding. Check32 separate
SciPy constrained projection solutions, noiseless inversion/energy1e-12,
23232 perturbations across the complete11³ ordered-two-domain grid and the
pointwise coordinate/W1 bounds. Gradient-check only the smooth carrier away
from projection boundaries; do not claim differentiability of optimal fitting.

Predeclare source stability probes: perturb mode mass by±epsilon for epsilon
1e-2,1e-4,1e-6,1e-8 in the50/50 two-mode case and symmetric three-mode case.
Record source W1, both carriers' distance, optimal split and intrinsic error.
Hard quantile branches and hard global split choices can jump across density
valleys/tied optima; retain those failures. If Q1 is unstable or loses task
information, it is not eligible for direct main training solely because one
synthetic example improves. Native train-only teacher sufficiency, differentiable
parameterization, matched generic latent/adaptation/exposure/compute/budget,
KITTI task evaluation, multiseeds and original wireless matrix remain required.
