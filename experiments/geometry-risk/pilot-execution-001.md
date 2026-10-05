# Full64 native risk pilot execution addendum001

2026-10-04; prelock before full64 observations/fitting. Eligibility evidence
commit7bc6c27 closes all6 original RGB endpoints and two-frame native engineering.
Independent native auditor a5ddbc5 passes260actualrows/complete sensors/calibration/
noise/posterior/535states and actualPIDs73911/68491terminal; local transferred
verifier0f59fa0 passesall260rows. Initial missing-native-yaw auditor failure is
preserved260fad2; model/data/measurement sources did not change. Sole complete
F6b parent remains77bc73021a139099e3d0f9c12e0fba8b17eb62f9b9c243eb8f0ad099bd6e8867.
F7 final weights must not replace it. OriginalF7/RGB actual process scopes are
terminal and fully closed. This is training-loss diagnostic calibration only,
not mainval/AP/allocation or inference-risk policy validation.

Exact32fit and next32out-of-fit IDs remain those already fixed in
native-engineering-inputs-001.json (SHA269ee72aafeadaeeb8e16d594f839d0a00b10390de35f5b903f62e0d45dd7430).
Out-of-fit means outside auxiliary ridge fitting only; all64 belong to the
perception-training set and cannot establish heldout perception generalization.
Engineering's260observations are NOT rows in the full64 fitting dataset.
Clarification: firsttwo SAMPLE IDs stay in the fixed32fit roster, but all pilot
intervention targets are newly measured with distinct noise draws. Same clean
sensor/features may recur, and the fact these two scenes were inspected during
engineering must be disclosed. No favorable replacement/exclusion or newscene
selection based on measuredlosses. LegalemptyGT/no-valid-geometry scenes stay
in observation accounting and invalidcoverage reporting.

Keep seed2801 PCG64, but continue from the SEALED engineering FINAL RNG state
in geometry-risk-native-engineering-001.json, not reset to engineering's initial
state. Independent replay must recover that state from all258engineering draws
before checking the pilot. Thus the repeated firsttwo scenes have fresh masked
noise rather than importing engineering observations. One continuous generator
through the64frames;128masked+onefull noise draws/frame,8320total received-only
loss passes including64clean gradients;519168000 attemptedcomplexuses. Retain
all actualnoise arrays, RNG states, native fixtures and every loss row. Same
CPUdrawthenmaskcastFP32, frozen535states, nooptimizer/teacher/depth/2Dhead,
deterministicnativecrop/padding/calibration/GT-at-head sequence, fixed62400layout
and all public groups as audited engineering. All attempts charged in full.
Require physicalfree>=12GiB and peak+2GiBmargin, preserve unique failure outputs.
Run one detachedprocess on sheng; no changes to sealed engineering/root source.
A new driver may reuse unchanged measure_frame/RiskObserver/RiskChannel helpers
and must include both helper and new driver source identities before/after each
frame. Commit implementation before actualexecution; no modified-old-result or
best-checkpoint initialization. No GPUreservation while idle.

Analysis details making the original pilot contract concrete before execution:
- Common complete-case group rows require positivefinite normalizedcodeenergy,
  positivevalid_count and finite entropy/depth_variance/coverage/gradient.
  Preserve every missing/invalidgroup and excludedcounts. Featureenergy is
  naturallog(code_energy_mean); geometryvector is entropy, rawdepth_variance,
  valid_count/candidate_count. No GT regionmask, featureselection or logdepth
  selected after outcomes. Gradient diagnostic is squared_gradient_sum; do not
  claim availability at inference or confuse gSigma g with mean damage.
- Target remains positivepart of the4-draw SIGNED mean native-loss increment,
  not mean of positive increments or true expected AP/lossdamage. Always retain
  all signedmeans,4individualdraws and samplevariance. Finite-sample clipping
  bias and four-draw variability must be disclosed.
- Five fixed ridge predictors of that target: energyonly; geometryonly;
  energy+geometry; gradientonly; shuffledgeometryonly. Fit standardizedfeatures
  on the first32 observations only, populationstd(ddof0), constantstd replaced
  by1; unpenalizedintercept, coefficientridge1.0, direct linear solve, no clipping
  predictions or tuning. Allmodels use EXACTLY the same complete-caserows.
- Shuffledgeometry uses PCG64 seed2802, one deterministic within-frame permutation
  among commoncompletecasegroups, whole3-vector moved together, frameorder64.
  Energy/gradient/targets stay in their owncells. Preserve permutations. Invalid
  cells are not relocated into the comparison support.
- Out-of-fit prediction MSE is computed within each frame then averaged across
  definedframes, rather than presenting32correlatedcells as independentframes.
  Also report per-frame Spearman versus signedmean, positivemean and sampled
  fluctuationvariance; n<3 or constantvectors yield undefined, with counts.
  Bootstrap1000 draws of the32OUT frames using PCG64seed2803; retain per-frame
  scores and resamplingindices, percentile2.5/97.5% intervals of definedframe
  means. Report undefinedframes explicitly; never turn them into zero/errorfree
  samples or silently change the frame bootstrap unit.

A complete independent rawnoise/input/publicgeometry/state/chronology audit and
actual terminal proof must precede any fitted result being trusted. If geometry
has no predictive evidence against controls, record that result and revise the
hypothesis before training an inferencepriority head. No claim of calibrated
uncertainty or scientific novelty follows from simply completing this pilot.
