# Geometric distortion requirements after the negative F9 primary

F9's true-correspondence gain coupling does not exceed either same-capacity
generic or shuffled control at the prelockedAWGN10endpoint. Preserve that
result. The next mechanism must change usable geometric information or an
operationally justified resource allocation, with matched task supervision,
resources and training. Descriptive small gains do not identify the cause.

For a rectified calibrated stereo rig, use normalized coordinates rather than
assuming equal principal points without checking:

\[
r=(u_L-c_L)/f_L-(u_R-c_R)/f_R=B/Z.
\]

For equal focal length and a corrected pixel disparityd, letA=fB>0, Z=A/d.
An error e in that corrected disparity gives the exact relation

\[
\widehat Z-Z=-\frac{Ae}{d(d+e)}.
\]

Ifd>0 and|e|≤ρd with0≤ρ<1, then

\[
|\widehat Z-Z|\le\frac{A}{d^2}\frac{|e|}{1-\rho},\qquad
(\widehat Z-Z)^2\le\frac{A^2}{d^4}\frac{e^2}{(1-\rho)^2}.
\]

This is a conditional algebraic bound. It requires a controlled relative-error
event; the complement must retain its probability and task cost. It is not a
rate-distortion lower bound, an AP theorem, a calibrated posterior or proof that
transmitting a cosine-correlation score achieves that error bound.

Stereo localization error depends on paired perturbations. If their centered
pixel errors areεL andεR, thenVar(εL−εR)=VarεL+VarεR−2Cov(εL,εR). Marginal
view uncertainty alone cannot specify disparity variance. Correlation of
correspondence probabilities or group gains is not measured physical error
covariance. A joint code needs a paired error/intervention measurement.

An unbounded additive Gaussian model for disparity creates another trap:
the perturbed disparity has positive density nearzero, and the integral of
1/(d+e)^2 diverges there. A global squared-depth expectation cannot follow
from the local derivative by assuming arbitrary Gaussian disparity errors.
Use an explicitly bounded operating range or a reliability/outage event and
charge its complement. Actual nonlinear feature noise is not assumed Gaussian
disparity noise; this mathematical example does not claim infinite error in
our implemented detector. Cost sampling range also does not by itself bound
the final box-regression output.

Likewise, σ²/a² effective noise is a coherent gain-inversion model, requiring
receiver gain information. F9's receiver does not invert private sender gains;
its learned codec receives changed content. A water-filling argument for pure
unequal protection cannot be applied to F9 without establishing the requisite
decoder model and charging any gain/mask/normalization signaling.

Priority remains complete main baselines. Before launching a new mechanism,
verify its actual calibrated geometry/paired error target on training data,
then test same-capacity generic/shuffled controls under complete main AP and
multiple seeds. The bound above motivates checks; it does not satisfy those
experiments or establish novelty. No post-result primary redefinition.
