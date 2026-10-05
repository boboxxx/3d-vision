# Full64 native risk pilot001: geometry does not yet identify mean damage

2026-10-04. Protocol8ea728b preceded implementation621113f and actualPID74257.
All64 fixed training scenes completed8320 native passes/519168000 attempted
complexuses. SoleF6b checkpoint77bc7302… stays fixed;535 states unchanged,
zero optimizer updates, no inference GT/teacher/depth/2D bypass. All2048 groups
have common valid support:1024 fit and1024 auxiliary check, no excluded cells.
The second32 scenes are unseen by the auxiliary ridge fit, but belong to the
perception training set; these are not main validation or AP results.

Full server audit independently replayed native images/GT/public calibration,
all retained noise arrays/RNG chronology, posterior/group arithmetic and states.
Two failed reference checks are preserved: CPU FP32 norm accumulation and
training class merges incorrectly assumed in TESTmode. Separately locked repairs
changed only the auditor, at original tolerances; all observations and sources
remained unchanged. Third full audit passes. Local raw metadata audit verifies
all8320 rows/64 reports; large arrays stay on sheng. Independent augmented least
squares/rank/frame-bootstrap audits pass both server and local runtimes.

Five prelocked predictors fit positivepart(four-draw signed mean) using first32
frames only, common support, train-only standardization and ridge1.0. Shuffled
geometry moves its whole three-vector within each frame using seed2802. Scores
are averaged over32 check frames, with1000 frame resamples seed2803. All32
frames define every metric. Point estimates and percentile95% intervals:

| Fixed predictor | Positive-target Spearman | Prediction MSE | Fluctuation-variance Spearman |
|---|---:|---:|---:|
| Code energy | .1055 [.0421,.1685] | .000443839 | .2031 [.1237,.2791] |
| Geometry | .0446 [-.0227,.1132] | .000451963 | .3699 [.3021,.4305] |
| Energy+geometry | .0881 [.0209,.1526] | .000440612 | .3485 [.2772,.4142] |
| Gradient diagnostic | .2954 [.2066,.3749] | .000475629 | .8834 [.8616,.9016] |
| Shuffled geometry | .0283 [-.0285,.0921] | .000465529 | .0100 [-.0601,.0767] |

Signed-target Spearman is .1045(energy), .0306(geometry), .0823(combined),
.1594(gradient), .0144(shuffled). Full confidence intervals, fit coefficients,
all predictions/per-frame metrics, raw targets and resampling indices are retained
in data/analysis and the64 reports. MSE intervals overlap substantially; marginal
interval overlap is not a paired significance test. Combined MSE is numerically
slightly smaller than energy-only, but this alone establishes no reliable gain.

The geometry-only positive correlation interval includes zero and its point
estimate is below energy-only. This pilot does not provide usable evidence to
train a geometry-as-mean-damage priority head. Its comparatively stronger
variance ordering motivates a different diagnostic question. Gradient's strong
variance ordering is consistent with g^T Sigma g describing first-order loss
variability; it is label-dependent and unavailable directly at inference.

The positive target itself has finite-draw clipping bias: even zero expected
linear loss change yields positive clipped means in expectation. See
[the target scope derivation](../../docs/four-draw-positive-damage-target.md).
The prelocked target was retained unchanged; it must not be relabeled as true
expected preventable loss or AP damage. A single full-noise sample per frame
also cannot estimate that expectation. Across all2048 masked groups,1103 signed
sample means are positive;45/64 full-noise draws increase loss. These are
descriptive observation counts, not independent-cell inference or effect proof.

Next: use fresh, public-roster training scenes and paired positive/negative noise
to separate symmetric mean increments from antisymmetric fluctuations, under a
new protocol. Do not launch an inference risk head/allocation policy on the
unsupported geometry mean-risk premise. Future policy still needs matched
channel adaptation, actual transmitted controls, mainval AP, fading and seeds.
