# Frozen predictor and permutation continuation

Before analysis implementation and all fresh8 measurements, clarify the
existing five-predictor comparison: use the exact old coefficients, training
means/scales and feature columns from pilot001. Recreate PCG64seed2802 and
verify its64 old within-frame permutations against the saved pilot result.
Continue that same generator over the8 new common-support rosters, applying
whole-vector geometry shuffles only for the shuffled-geometry predictor.
No new fitting or score sign selection. Invalid cells remain outside every
comparison. Zero/constant vectors produce undefined rank correlations.

Bootstrap exactly1000×8 frame indices withPCG64seed2805; average defined
frame correlations, retain undefined counts and percentile2.5/97.5 intervals.
No target or new feature changes. This deterministic continuation resolves an
implementation detail absent from the original paired-noise protocol.
