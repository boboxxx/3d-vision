# Fresh8 paired-noise result

Complete native execution and independent audit passed on sheng; all8 reports
and8208 raw rows passed local transferred verification. Five frozen predictors,
all per-frame ranks and1000 eight-frame bootstrap replicates independently
recomputed on both hosts. No failed attempt, refit, excluded frame or undefined
rank in this run. All256 groups satisfy the common support. PID161285 exited;
native manifest ended2026-10-04 16:49:04UTC. Its audit-pending state is immutable;
separate successful audit files close the run.

Fixed F6b535-state model, no optimizer/parameter gradients. Exactly4104 independent
noise draws,8208 attempts,512179200 complex uses; actual transmit energy
512179204.0078125. Peak allocated5.174890GiB, reserved6.837891GiB; physical
free before24152899584bytes. Same public32 groups and62400 complex uses each
attempt, all signs retained. Protocol2a23f8b and clarification03dd2ab precede
implementationbfe7999 and CPU evidence7df425c.

For each region,16 pairs give q=(delta+ + delta-)/2 and s=(delta+ - delta-)/2.
Under symmetric noise E[q] equals the expected finite-noise loss increment.
The signed sample mean is the relevant diagnostic; clipping it remains a
biased secondary quantity. Old32-frame fitted coefficients and feature scaling
were frozen; no fresh8 fit. Table entries are means of8 within-frame Spearman
coefficients with percentile95% intervals from the fixed frame bootstrap.

| Frozen predictor | Signed mean q | Positive part of mean q | Sample variance of s |
|---|---:|---:|---:|
| Energy | −.1165 [−.3083,.0992] | −.0639 [−.2539,.1367] | .2709 [.1561,.3552] |
| Geometry | −.0598 [−.1785,.0528] | −.0267 [−.1559,.0830] | .4512 [.3746,.5342] |
| Energy + geometry | −.0942 [−.2341,.0723] | −.0468 [−.1863,.1040] | .4601 [.4020,.5296] |
| Gradient diagnostic | .0509 [−.0990,.1900] | .1862 [−.0114,.3691] | .9601 [.9480,.9724] |
| Shuffled geometry | −.0088 [−.0842,.0804] | −.0052 [−.0912,.0795] | .0308 [−.1065,.1638] |

This small follow-up strengthens the distinction between predicting variability
and predicting mean damage. It supplies no positive mean-damage ranking evidence
for the current geometry score. Do not use the variance correlation as proof
that allocating power by geometry minimizes expected native loss or improves AP.
Intervals crossing zero are uncertainty, not proof of zero effect; the experiment
does not rule out other geometry representations or calibrated policies.

Strong variance ranking also does not validate a quantitative linear model.
Across512 pairs within each frame, RMS(s−g^T e)/RMS(s) ranges .6907–1.6731;
five of8 ratios exceed1. This shows substantial finite-noise residuals. Monte Carlo
standard errors of meanq are retained for every region, and negative signed
means remain in the results. No fitted Hessian or AP theorem follows.

All8 scenes are fresh relative to prior64 diagnostics but remain perception
training examples; this is not held-out perception generalization. Eight scenes,
one checkpoint/noise seed, coarse fixed geometry features, native loss rather
than AP, and labels required for the gradient diagnostic bound every inference.
Earlier results informed this experiment; it remains exploratory.

Decision: finish this bounded diagnostic and strengthen the uniform task model
with the separately prelocked F8 matched codec-only/joint encoder comparison.
Future policy learning should test actual benefit under equal resource changes,
including control costs, with AP as the endpoint. Do not train a mean-risk head
merely because geometry correlates with noise variability.

Evidence: `data/runs/geometry-risk-antithetic-001.json`, the native/statistics/
statistics-local/transferred audits under `data/provenance`, complete raw rows
under `data/engineering/geometry-risk-antithetic-001-records`, frozen-predictor
and descriptive JSON plus bootstrap indices under `data/analysis`. Large
sensor/feature/noise arrays were independently replayed on sheng only. Figure:
`to_human/plots/geometry-risk-antithetic-001-2026-10-04.png` (visually checked).
