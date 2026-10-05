# Fixed exploratory codec learning check F0

Locked before training, 2026-10-03. This is exploration, not an ICC result.
Full clean references are independently audited: Stereo-RCNN Car moderate
3D AP_R40 IoU0.7=34.0749706%; LIGA=67.7238123%, all3769 validation frames.

Draw372 holdout frames solely from the official3712 training IDs by the
smallest SHA256(`geocomm-codec-tuning-v1|frame_id`); retain original order in
both folds. Train3340, holdout372, no overlap with main3769 validation IDs.
Use only original training infos to generate these folds. The released detector
checkpoint was pretrained on the original training set, including this holdout;
the fold isolates newly trained communication modules, not original detector
pretraining. Keep final validation for frozen scientific comparisons.

F0: seed17, one complete3340-step epoch, batch1, AdamW LR0.001 constant,
weight decay0.0001, gradient norm clipping10, abort before nonfinite update.
No LR warm-up or decay for this one-epoch check. Codec/student/sensitivity
parameters train; original detector parameters and BN statistics are frozen.
Original image feature teacher and LiDAR imitation teacher remain training-only.
All native3D/depth/2D/imitation losses remain, plus posteriorCE0.1,
normalized feature distillation0.1, sensitivitySmoothL1 0.1. Native sensitivity
target excludes these auxiliary losses. No second-order graph.

First run uses uniform power and the complete student/sensitivity architecture.
This retains the importance head/supervision for later matched-capacity controls.
Complex width4, geometry stride(8,4,4), appearance stride4, AWGN trainSNR[-5,20],
holdout SNR10. Actual per-frame symbol/energy counts are logged, accounting for
variable original image width. No GT or LiDAR reaches evaluation forward.
The full receiver, original72 downsampled depth bins,320-height crop and native
evaluator remain. Posterior supervision includes measured edge half-bins of
the released2–59.6m range and excludes zero-filled missing depth.

Evaluate only the372 training holdout at epoch1. Save checkpoint, raw losses,
pre-clip gradient norms, predictions, link accounting and native AP. No learning
or AP improvement claim before actual updates/held-out results. Negative results
are retained. Any uniform/geometry/task/geometry_task exploratory comparison
uses this identical fold, one-epoch budget and same extra supervision. Future
confirmatory warm-up/joint budgets and seeds remain to be locked after learning
is established; do not compare this one epoch to a longer trained variant.

## F0b paired allocation check (locked before its launch)

Conditional on F0 completing with finite updates and auditable holdout outputs,
run geometry_task from the same released detector initialization and seed17,
not from the trained uniform checkpoint. Config differs only in allocation;
the fold, one3340-step epoch, optimizer, noise training distribution, student,
sensitivity supervision and all native losses are identical. Evaluate372
holdout frames at10dB with the same deterministic inference seed path. Keep
both results even if geometry_task loses. This remains exploratory.

F0 itself records the original protocol SHA in its pre-run manifest; this
appendix was added while F0 was already running and does not rewrite its
original scientific status or training budget. Commit F0b protocol/config
before launching it. Broader no-sensitivity/geometry-only/task-only controls,
noise repetitions and confirmatory budgets remain required.
