# Receiver uncertainty: prior work and limits of a proposed intervention

Primary sources checked on2026-10-05:

* [Cai et al., Probabilistic Modeling of Disparity Uncertainty for Robust and Efficient Stereo Matching, arXiv2412.18703v2](https://arxiv.org/abs/2412.18703v2),2025 revision. The abstract explicitly uses Bayes risk to measure data and model uncertainty, with four stereo benchmarks. Only the abstract/metadata have been read here; no claim about its full architecture or exact correspondence to this project.
* [Mehltretter, Joint Estimation of Depth and Its Uncertainty from Stereo Images Using Bayesian Deep Learning, ISPRS Annals2022](https://doi.org/10.5194/isprs-annals-V-2-2022-69-2022). The primary abstract describes jointly estimating disparity and aleatoric/epistemic uncertainty using a Bayesian network and variational inference on three datasets. Full paper not read in this check.

Generic Bayesian uncertainty or geometry-aware stereo is established territory.
The proposed experiment concerns a different, narrower variable: the posterior
of three ordered, already transmitted latent depth nodes given actual noisy
complex symbols and the main contract's perfect receiver CSI. A public
finite-grid prior is explicitly a model assumption, not learned KITTI depth
calibration. Efficient order-constrained message passing, conditional means
and squared-error optimality are standard inference principles; no new general
Bayesian theorem is claimed. A short search did not establish uniqueness or
absence of closer semantic-communication work. Novelty remains unproven.

Any successful physical synthetic check would establish receiver mathematics
under the declared prior. It would not establish AP, actual object depth error,
scene uncertainty calibration, source lightness, adaptive resource allocation
or superiority to the original RGB baseline. Existing84 primary validation
endpoints remain unchanged; receiver interventions require separately fixed
full validation and actual receiver cost measurements.
