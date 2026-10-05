# Ordered received-node inference under an explicitly declared finite-grid prior

The mathematical equal-mass barycenters of a distribution on an increasing
depth axis satisfy mu1<=mu2<=mu3. Their separate phase symbols preserve slot
identity; noise can invert received phase coordinates. This ordering is a
representation constraint, not the stereo pixel-correspondence ordering
assumption and not a constraint that objects are sorted across an image.
The frozen float32 implementation may incur roundoff near ties; no claim
about all actual stored transmitter nodes is made without checking them.

Let g_m=m/(M-1), theta_m=(2g_m-1)pi/2, with the public production M289, and
define a deliberately assumed discrete prior uniform over every index triple
a<=b<=c, including ties. It is not the empirical distribution of trained
KITTI latent nodes, or a calibrated distribution of actual object depths.
The evidence below can validate inference for this prior only.

For ZF-observed r_j=e^{i theta_j}+n_j/h_j, N0=q and actual perfect receiver CSI
h_j, the likelihood terms depending on the candidate phase are

\[
 \ell_j(m)=2|h_j|^2\operatorname{Re}(r_je^{-i\theta_m})/q.
\]

This follows by expanding the proper complex Gaussian baseband density
exp(-|y_j-h_j e^{i theta_m}|^2/q). For AWGN set h_j=1. It uses raw received
amplitude and actual CSI; it does not use realized noise, clean node means,
source probabilities, transmitter feature norms or GT. If CSI is deliberately
ignored in a Rayleigh ablation, the already derived CN quotient density yields
ell_j(m)=-2 log(q+|r_j-e^{i theta_m}|^2). Using average Rayleigh variance in a
Gaussian likelihood would be a different, misspecified model.

With L_j(m)=exp ell_j(m), forward messages A1=L1, A2(m)=L2(m)sum_{u<=m}A1(u),
A3(m)=L3(m)sum_{u<=m}A2(u); backward B3=1, B2(m)=sum_{u>=m}L3(u),
B1(m)=sum_{u>=m}L2(u)B2(u). Z=sum_m A3(m), and pi_j(m)=A_j(m)B_j(m)/Z.
Inclusive cumulative sums implement the declared ties exactly. Log-domain
cumulative logsumexp avoids numerical underflow when noisy slot modes disagree.
The exact discrete complexity is O(3M), instead of enumerating O(M^3) triples.
This is ordinary order-constrained sum-product inference, not a new general
probabilistic algorithm or a new Bayesian optimality theorem.

Recover the frozen Gaussian basis using its conditional expectation:

\[
 \bar w_j(z)=\frac13\sum_m\pi_j(m)
 \exp[-(z-(z_{lo}+Lg_m))^2/(2\sigma^2)].
\]

Then C_hat(z)=sum_j bar_w_j(z) v_hat_j, with the same noisy received features,
frozen cost decoder and fixed learned global sigma. It differs from placing
the basis only at the posterior mean. Conditional expectation minimizes
squared field error under the declared prior when the coefficients are fixed
and known, by expanding E[(C-a)^2|r,h]=Var(C|r,h)+(E[C|r,h]-a)^2.
Actual received features may correlate with node depths and contain noise;
ignoring that correlation is a factorization assumption. The implemented full
receiver is therefore not claimed to be the optimal full task posterior.

Same transmitted19 complex symbols and Es19/site; no source or packet change,
no added model parameters, no transmitted uncertainty label. The perfect CSI
assumption is the already fixed main Rayleigh contract. Its additional
receiver arithmetic/memory/latency must be measured, not described as free.
Identity mode delegates to the frozen decoder bit-for-bit. Continuous source
nodes are approximated by289 grid points; synthetic prior checks use33 points
and prove that declared33-grid model, not discretization accuracy for arbitrary
continuous nodes or actual289-grid AP. Current full84 validation is unchanged.

Generic uncertainty/Bayes-risk stereo has prior work; see
[background and verified primary-source scope](../../literature/ordered-channel-receiver-background-2026-10-05.md).
Physical-prior MSE is not 3D object error, AP, rate-distortion theory,
uncertainty calibration on KITTI, adaptive source resource allocation or
publication-level novelty. Receiver interventions require separately locked
whole validation, actual resources, and source/receiver evidence.
