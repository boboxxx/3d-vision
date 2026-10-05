# Phase coordinates under the frozen physical channel

The result below is a calculation for codec004's explicit position symbols. It
applies to the continuum channel/decoder, conditional on the transmitted theta.
It does not describe the learned source distribution or the final detector.

Write L=57.6, mu=2+L/2+L theta/pi, and q=10^(-SNR/10). Positive sphere
projection preserves angle except for the near-zero fallback documented in the
protocol. Isotropy allows rotating x=exp(i theta) to1. Define phi=arg(1+w),
a=cos(phi), and b=sqrt(q+sin(phi)^2).

For AWGN w~CN(0,q), polar integration gives

\[
p_A(\phi)=\int_0^\infty \frac{r}{\pi q}
 \exp[-(r^2+1-2r\cos\phi)/q]\,dr
=\frac{e^{-1/q}}{2\pi}
+\frac{a}{2\sqrt{\pi q}}e^{-\sin^2\phi/q}
 \operatorname{erfc}(-a/\sqrt q).
\]

For a<=0 the implementation factors out exp(-1/q) and uses erfcx to avoid
subtracting large terms. The declared SNR range does not require an asymptotic
approximation.

For Rayleigh equalization, change variables n=w*h in the independent joint
complex Gaussian density. The real Jacobian is |h|^2; hence

\[
f_W(w)=\int_{\mathbb C}\frac{|h|^2}{\pi^2q}
 e^{-(1+|w|^2/q)|h|^2}\,d^2h
=\frac{q}{\pi(q+|w|^2)^2}.
\]

Its phase after shifting by1 is

\[
p_R(\phi)=\frac q\pi\int_0^\infty
 \frac{r}{(q+r^2+1-2ra)^2}\,dr
=\frac{q}{2\pi b^2}
 \left[1+\frac ab\left(\frac\pi2+\arctan\frac ab\right)\right].
\]

To evaluate the radial integral set u=r-a, so the denominator is (u^2+b^2)^2.
Integrating the u term and the a term from -a to infinity yields the expression.
Numerical implementation uses atan2(b,-a) for the bracketed angle.

Let W(t) wrap to[-pi,pi) and C(t)=clip(t,-pi/2,pi/2). The exact conditional
coordinate moment and clipping probability are

\[
D(\theta,q)=\left(\frac L\pi\right)^2
 \int_{-\pi}^{\pi}[C(W(\theta+\phi))-\theta]^2p(\phi)\,d\phi,
\quad
P_c(\theta,q)=\int_{-\pi}^{\pi}
 \mathbf1_{|W(\theta+\phi)|>\pi/2}p(\phi)\,d\phi.
\]

These expressions include phase wrapping and clipping, not merely small-noise
linearization. Always0<=D<=L^2. In contrast E[1/|h|^2] diverges, so averaging
the conditional linearized variance q/(2|h|^2) does not give this moment.

For the actual fallback event max(|Re y|,|Im y|)<=eps32, both rotated AWGN and
shifted complex-ratio densities are bounded above by1/(pi*q). The square's area
is4eps32^2. Thus its probability is at most4eps32^2/(pi*q), and the absolute
change of the bounded MSE is at most L^2 times that bound. This is a loose
continuum fallback bound, not a bound on rounding throughout the codec.

Complex Gaussian quotient statistics are established prior mathematics. Gu's
2020 NIST paper concerns a zero-mean numerator and a nonzero-mean denominator;
it is background, not a citation for this particular zero-mean fading case or
our clipped coordinate formula. The calculation here is explicitly derived
and independently checked for the frozen codec. No new general distribution
theorem or task rate-distortion theorem is claimed.

Verified background: [Gu, NIST,2020](https://www.nist.gov/publications/quotient-centralized-and-non-centralized-complex-gaussian-random-variables),
[DOI10.6028/jres.125.030](https://doi.org/10.6028/jres.125.030).
