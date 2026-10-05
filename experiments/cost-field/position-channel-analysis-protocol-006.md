# Exploratory packet coordinate sensitivity, not object depth error

Keep full training004, inference005 and primary P-B AP untouched. Analyze only
the already sealed first256 G updates of seed17, all1560 sites/three position
symbols per frame, not a favorable node/frame subset. Use immutable actual
report snapshot004-003 and identical saved update rows/artifacts/checkpoints.
Rerun complete record checks and match their sealed prefix JSONL/ledger hashes.

Independently recover packet coordinates from source unit-phase symbols and
raw received symbols, using atan2, the locked half-circle clamp, known2..59.6
axis and the same zero-receive fallback. Collect all squared/absolute signed
coordinate displacements, clipping and reversed adjacent-node order. Group
only by AWGN/Rayleigh and predefined SNR intervals[6,8),[8,10),...,[16,18).
Report every interval and counts. No GT, AP, posterior calibration, numerical
gradient or full training-seed conclusion follows from these values.

For an interior node, small complex AWGN gives Var(delta_theta) approximately
N0/(2*Eposition); hence Var(delta_coordinate) approximately
(depth_span/pi)^2/(2*Eposition*gamma). For unit position energy and57.6m span,
the high-SNR interior prediction at10dB is about4.10m RMS. This is only a local
linearization: finite-SNR angular wrapping/clamping and boundary effects need
the actual whole-prefix measurement. It is not a detector localization bound.

Conditioned on Rayleigh h, the linearized expression also divides by|h|^2.
Its fading average is divergent; the real bounded half-circle decoder has
coordinate squared error bounded by depth_span^2. Do not call real decoded
error infinite, or present the linearization as exact rate-distortion theory.

Scientific question: does explicit metric-coordinate transport itself incur
substantial channel displacement at the fixed budget? This may explain a
future result and motivate a separately locked resource study; it does not
justify changing current energy allocation, kernels, epochs or endpoints.
