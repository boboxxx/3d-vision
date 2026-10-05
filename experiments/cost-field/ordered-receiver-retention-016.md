# Preserve every paired risk array, without changing receiver015 or criteria

Prelock before executing016 or inspecting first015 risk outcomes. Static review
found015/check.py holds paired per-trial errors only in memory and writes their
summaries; protocol015 explicitly says retain those errors. Keep the original
015 code, actual run and all outputs. They can establish summary/algorithm
checks, but not persisted per-trial evidence. Do not claim the retention
requirement is complete from the summary report alone.

Isolated016/check.py imports unchanged015 receiver and repeats all original
enumeration, integration, physical source draws,20 conditions,65536 samples,
comparators, prior, field coefficients, likelihoods and fixed8SE+1e-12 criteria.
The only numerical-output change is saving all five coordinate and six field
per-trial error arrays before each condition's assertions, into exclusive
output-stem.paired/channel-snr.npz. Record full file SHA, array count and paired
trial count in the condition summary; preserve partial artifacts if a check
fails. Final state explicitly identifies retained paired errors. No criterion,
source, decoder or prediction change. No AP or actual KITTI prior claim.

Native and local complete evidence additionally requires re-reading every saved
array, finite float64 shape65536, complete named arrays, report-bound SHA and
independent re-computation of every reported MSE and paired standard error.
All20 files and both actual exits0 are required. Files remain small research
data in their native/local stores; no need to duplicate native115MiB collection
locally. Existing actual full-train/native/local/validation gates are unchanged.
