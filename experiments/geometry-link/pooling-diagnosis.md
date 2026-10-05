# F4 follow-up: fixed pooling diagnosis, no training or wireless claim

F4's fixed3340 reconstruction updates reduced identity cost/appNMSE to
.6102321605/.1500258911 and approached the simple pool/interpolation reference,
but both identity andAWGN10 full372 native3D AP were0. This motivates examining
pooling before further optimizing the allocation proxy. This is a newly locked
exploratory diagnostic, not a retrospectively selected confirmatory ablation.

Use the exact fixed finalF4 checkpoint
984771355f145441f785a8bca5ddde2762fec4925d504298b7fb9abe8c3f57f0,
the same frozen student/native downstream, sensor-only inference, seed17 and
all ordered372 training-only heldout frames. No training, augmentation,
checkpoint selection, mainval or extra epochs. Disable communication explicitly;
load every519 required non-link state strictly from the549-state checkpoint,
record the30 intentionally unused link states and verify their names. No
physical rate, energy, communication performance, oracle or AP ceiling claim.

Run all five predetermined conditions (including the new wrapper's no-op
control); never choose a condition after examining its AP:

| Condition | Raw cost operation | Appearance operation |
|---|---|---|
| control | unchanged | unchanged |
| cost_only | adaptive mean pool(8,4,4), trilinear interpolate | unchanged |
| appearance_only | unchanged | adaptive mean pool(4,4), bilinear interpolate |
| both | pool(8,4,4), trilinear interpolate | pool(4,4), bilinear interpolate |
| depth_preserved | pool(1,4,4), trilinear interpolate | unchanged |

Use ceil-sized grids identical to GridPool, align_corners=False, original
native feature shapes restored. Only the actual raw cost before native dres0
and the left student's appearance may be transformed; stereo feature inputs,
calibration, downstream parameters and metadata remain fixed. Record the
executed operation and shapes once per frame, student2/raw-cost1/module calls,
zero original teacher calls, whole model-state equality before/after inference,
complete AP/prediction/GT/calibration/ID audits and source hashes at closure.
Report all conditions, differences vs the predetermined wrapper control and
limitations. Global cost NMSE is not a task damage measure, and this does not
establish a sufficient representation, a unique failure cause or a new method.

Interpretation gate: if cost pooling itself destroys AP, prioritize depth/spatial
information preservation and separately budgeted receiver adaptation; if pooled
features retain useful AP while the trained F4 codec does not, inspect residual
codec distortion/task adaptation. Appearance-only and depth-preserved conditions
help localize sensitivity but cannot uniquely attribute the complete cause.
Any subsequent training or changed architecture requires a separate precommitted
protocol with charged exposure/budget. Schedule GPU work around the live original
83-epoch cycle; validate memory/resource coexistence before executing this suite.
