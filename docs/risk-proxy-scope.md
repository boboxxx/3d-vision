# What the allocation proxy can justify

Current implementation uses positive weights from depth posterior variance and
a predicted squared native-task gradient. This is an empirical weighting rule.
There are three different quantities that must remain distinct:

1. Intrinsic depth ambiguity given a clean stereo pair.
2. Error in transmitted/decoded features induced by the channel.
3. Change in downstream detection performance caused by that error.

A large value of (1) does not imply that more communication can remove it.
The posterior may be uncertain because of occlusion, weak texture or a wrong
model. Allocation must be validated against **preventable channel damage**.

Let a smooth detector output be f(z) and a zero-mean channel-induced feature
perturbation be e with covariance Sigma. A local linear approximation gives
E||f(z+e)-f(z)||² ≈ tr(J Sigma Jᵀ), where J is the output Jacobian. This requires
small perturbations and an appropriate covariance; it is not a formula for AP.
With independent perturbation groups it separates into group contributions.
For a smooth scalar training loss L, the corresponding second-order expected
loss change is approximately ½tr(H Sigma), with H its Hessian. A squared loss
gradient instead approximates variability of the first-order loss perturbation,
gᵀ Sigma g. These objectives are different. We have not computed or validated
a native-loss Hessian model, and the learned sensitivity has no AP guarantee.

Depth variance multiplied by squared task gradients is not derived from either
expression without a model connecting posterior uncertainty to channel-induced
latent error. The implemented nonlinear posterior has the capacity to depend on
both views; that engineering property does not establish calibrated uncertainty
or the required covariance relationship.

The implemented optimizer solves only this explicit scalar surrogate:

    minimize Σ_i a_i / (1 + gamma p_i)
    subject to p_i >= 0 and Σ_i p_i = N.

For positive a_i and gamma, the objective is convex; stationarity yields

    p_i = max(0, (sqrt(a_i gamma / lambda) - 1) / gamma),

with lambda chosen to satisfy the budget. This establishes optimality for the
surrogate under its assumptions. It does not establish optimal transmission
for correlated learned features, non-Gaussian errors, noisy estimated fading,
or discontinuous AP. In the existing receiver, the learned decoder is not an
explicit scalar MMSE estimator supplied with each private power coefficient.
Thus its feature-error law must be measured rather than assumed to equal
1/(1+gamma p_i). Retain the algorithm as an allocation heuristic until then.

After useful representations are obtained, record teacher-free posterior
calibration and observed damage from independent channel perturbations on the
training-only holdout. Compare depth variance, entropy, task sensitivity, their
product and a matched-capacity generic scorer at identical uses/energy. Test
whether the interaction improves predictive damage ordering and actual 3D AP;
correlation alone does not establish the final gain. Any fitted calibration or
selection stays within the training-only fold. Main validation remains for the
locked experiment.

Relevant established task-risk and digital-reliability approaches are recorded
in [the primary-source audit](../literature/survey.md). This note is a mathematical
scope analysis of our proposed proxy, not a new rate-distortion theorem.
