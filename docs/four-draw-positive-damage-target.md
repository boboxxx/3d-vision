# Finite-draw positive damage and gradient variability

The prelocked pilot target is max(mean of four signed loss increments,0).
It differs from the positive part of the true expected loss increment.
This note interprets the existing target; it does not change fitting or endpoints.

Within a smooth local loss region, write one masked first-order increment as
g_j^T e_j. The pilot's complex CN(0,.1) intervention has variance .05 in each
real I/Q component. With independent draws and frozen clean gradient,

    Var(one first-order increment) = .05 ||g_j||²
    Var(mean of four)             = .0125 ||g_j||²

In this linear Gaussian approximation, the true signed mean is zero, but

    E[max(four-draw mean,0)] = sqrt(.0125 ||g_j||² / (2 pi)).

Consequently, larger gradients can predict the sampled positive target even
without positive expected mean damage. This is clipping bias coupled to noise
variability. The nonlinear detector, target assignment and finite perturbation
need not satisfy the local approximation globally; the formula is a conditional
diagnostic explanation, not an empirical result or AP theorem.

Report signed means, the four raw draws and sampled variance alongside positive
targets. A geometry or gradient correlation with the latter alone does not
identify removable expected loss or an effective allocation. Additional signed-
mean estimation and actual equal-resource allocation AP are needed for that
claim. Native label-dependent gradients remain training-only diagnostics.
