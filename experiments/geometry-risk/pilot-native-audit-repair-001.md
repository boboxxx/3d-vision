# Independent norm reduction repair001

2026-10-04, after full64 observations, before any risk fit. First CPU audit
failed comparing the recorded CUDA symbol-gradient norm against NumPy's
float32 norm. Preserve the original verifier identity, complete failure log,
manifest and 64-frame diagnostic. No native measurements or sources change.

Read-only diagnosis of the same retained float32 arrays finds CPU float32
reductions systematically underestimating the norm. Double precision norm
agrees with the recorded CUDA result within about 1e-7 relative error in the
failed cases. The CPU audit should accumulate squared float32 values in
float64 before taking their norm; retaining float32 for this 124800-component
reference check unnecessarily introduces reduction error.

Change ONLY the independent verifier's reference norm to float64 accumulation.
Keep the original rel_tol=3e-6 and abs_tol=1e-6, all 64 frames and every other
assertion. Rerun the complete noise/input/posterior/state/chronology audit into
its still-absent unique output; use a new log, preserving attempt001. Statistical
analysis remains gated on full independent audit and actual terminal proof.
This is an audit numerical-reference repair, not a successful first attempt.
