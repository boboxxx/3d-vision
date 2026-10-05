# F9 four-arm native engineering closes

From the same complete F8 joint parent, U/G/P/S each finished six native KITTI
updates, fresh AdamW and one empty-GT/background-only frame. No AP was measured.
The unique detached controller PID255389 exited; independently prepared post-terminal
closure succeeds. Complete27 transferred artifacts and24 raw training records
pass local verification. All old seven-tree source identities remain sealed.

| Arm | Selected tensors | Fixed states | Actual updates per tensor | Peak reserved GiB |
|---|---:|---:|---:|---:|
| U uniform |51|488|6|7.33203125|
| G generic |55|484|6|7.333984375|
| P true epipolar |55|484|6|7.41015625|
| S shuffled epipolar |55|484|6|7.41015625|

All539 saved states, exact535-state parent loading before head initialization,
four extra public-initialized states, actual optimizer counts/moments and frozen
states are audited onsheng. U fixes the additional four head states; all arms
fix the484 detector states. All selected tensors receive finite learning signals
across the six steps. Initial hidden-head gradients are exactly zero as expected
from the zero final layer; later gradients/moments are present. No zero-gradient
sample or empty-GT frame is dropped.

Actual augmented left/right input, GT, calibration, transform and dedicated noise
streams pair across all four arms. Their first loss is exactly1.8338221311569214,
all first gains are1, and all first energy records match the uniform baseline.
Every attempt carries62400 complex symbols,49920 stereo plus12480 appearance,
no header/pilot symbols for these AWGN conditions. All64 measured group energies
plus appearance sum to actual full transmitted energy. No gain/matching map,
clean feature or normalization factor reaches the inherited receiver decoder.
GT enters only at the native3D head; teacher/depth/2D branches are never executed.
Physical free memory and peak-reserve-plus2GiB margin satisfy the prelocked gates.

A passive profile on each first actual sender forward records parameter counts,
operator trace and supported-operator FLOPs lower bound. U/G/P/S record about
5.4289/5.4291/5.4909/5.4909 billion supported operations. This profiler omits
unsupported operations and does not replace a complete analytical FLOPs count.
Cold sender wall times including audits/profiling are about0.5803/0.4931/0.5167/
0.5374 seconds. They are local engineering observations, not warm deployment
latency or speed comparisons. Complete traces remain onsheng and were rehashed
there; the local transfer contains their evidence, not those large traces.

Protocold8b3532 precedes implementation7ca9715/bde2089/4641563 and launch17b02c2.
Full server closure4e6517d and independent local verifier3b85fde were prepared
before execution of their checks. Fifteen CPU check families passed locally
andsheng before the unique engineering launch. No failed native attempt or
protocol repair occurred. Engineering weights are excluded from formal training.

Evidence: `data/provenance/stereo-epipolar-native-sanity-001-closure.json` and
`stereo-epipolar-native-sanity-001-local-verification-001.json`. Next: complete
539-state fixed receiver/evaluation integration, preserving all current core
source identities; then a unique formal3340-step/four-arm cycle and all16 fixed
372-frame endpoints. Geometry-specific benefit is unproven until P−G and P−S
AP comparisons pass. Main validation, original scenarios and multiple seeds
remain mandatory for the research claim.
