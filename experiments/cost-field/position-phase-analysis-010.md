# Coordinate reliability depends on phase location and physical channel

The exact continuum conditional moment passed all130 predefined conditions on
both local CPU and sheng CPU, with all26 phase PDFs normalized and independently
checked against radial quadrature. Each host generated27262976 physical channels
and136314880 coordinate observations. The largest Monte Carlo MSE deviation is
2.650 sample standard errors; all fixed8SE criteria passed. Independent runtime
versions differ (NumPy2.4.4/SciPy1.17.1 versus1.26.3/1.14.1); largest cross-host
exact MSE difference is2.85e-14 m^2, and MC MSE difference8.53e-14 m^2. This is
verification of these scalar calculations, not all-model CPU/CUDA equivalence.

At10dB, the fixed source phases give:

| Source phase | AWGN RMSE, m | Rayleigh/ZF RMSE, m | AWGN clip probability | Rayleigh clip probability |
| --- | ---: | ---: | ---: | ---: |
| 0, interval midpoint | 4.219 | 8.060 | 0.000003872 | 0.023269 |
| +/-pi/3 | 4.159 | 8.301 | 0.012674 | 0.077423 |
| +/-pi/2, interval boundary | 2.985 | 9.022 | 0.500000 | 0.500000 |

Negative phases have equal analytic moments by symmetry; all were simulated and
reported in the full130-row CSV. Figure lines show the three unique exact curves
and circles the positive-phase simulations; no angles or SNRs were fitted.

![Exact clipped phase moments](/Users/chen/Documents/ChatGPT/paper6/to_human/packet-position-phase-theory-010.png)

At an AWGN boundary, clipping mostly removes the outward small perturbation;
at10dB the RMSE is lower than at the midpoint. Under Rayleigh equalization,
large phase excursions retain a substantial opposite-boundary error. For
theta=pi/2 and phi in[pi/2,pi], phase wrapping followed by clipping maps the
coordinate to the other interval endpoint, an error of exactly57.6m in the
continuum model. This event has probability half the midpoint clipping
probability, about0.011634 at10dB Rayleigh, and contributes about38.60m^2 to the
81.40m^2 boundary MSE. This endpoint-tail decomposition is a post-result
analytic interpretation, not a separately prelocked experimental endpoint.
The same boundary clipping probability of0.5 for both channels therefore does
not mean equal reliability. Do not average the divergent linearized1/|h|^2
variance or remove genuine deep fades to obtain a favorable curve.

Frozen float32 codec004 also has a near-zero fallback. The loose uniform
density bound limits its continuum MSE effect to3.79e-9m^2 over this grid; no
fallback occurred in these draws. The bound excludes all other rounding errors.
The first009 implementation exited1 on NumPy float32 JSON serialization. Its
source/log and failure record are retained; separately prelocked010 changes
only the bound scalar's serialization, with no formula or threshold changes.

This mechanism explains a precision cost of the explicit position symbols,
not object localization error or detection AP. The actual learned distribution
of packet coordinates is not any of the five fixed source phases; the previous
006 actual-prefix measurements use different coordinates and continuous SNR.
They must not be compared as identical conditions. Features, spatial
interpolation, learned bandwidth and task head can mitigate or amplify packet
error. Existing negative internal-split primary results remain negative; this
analysis establishes no new-method improvement or sufficient representation.

Current authorized formal scope is seed17,44544 updates, four fixedfinalepoch3
models and complete3769 validation, followed by original wireless/baseline and
actual resource controls. Historical three-seed133632 in analysis006 is
superseded by single-seed-scope007. No multi-seed significance, AP result,
Shannon rate-distortion bound or new Gaussian quotient theorem is claimed.

Evidence: position-phase-theory-protocol-009.md, position-phase-derivation-009.md,
position-phase-serialization-repair-010.md,
data/provenance/position-phase-theory-two-host-closure-010.json and both complete
local/sheng010 result directories. Gaussian-ratio background was verified from
[Gu's NIST2020 publication](https://www.nist.gov/publications/quotient-centralized-and-non-centralized-complex-gaussian-random-variables);
our zero-mean fading and clipped-coordinate calculation is explicitly derived,
not attributed to that paper's nonzero-mean denominator result.

Exploratory fixed-checkpoint observation012: all G/P/S finalepoch3 states were
safely read and already hash-verified. Their shared scalar Gaussian widths are
9.501m/7.474m/34.667m respectively. S is a spatial roll of the native depth guide
by half the pooled height and width, not a different sigma initialization. The
mismatched-guide S checkpoint has a broader learned field basis. This is a
parameter observation from fixed checkpoints, not AP selection, proof of why
training chose that width, calibrated depth uncertainty or geometry benefit.
Separate evidence: cost-field-fixed-epoch3-bandwidth-observation-012.json.
