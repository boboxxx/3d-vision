# Complete LIGA-Stereo clean detector reference

Official upstream aee3731a24a0ab1667e633e520cc89be2f135272, full released
architecture, 320-height crop, original 72 downsampled depth bins and complete
receiver. Author epoch53 checkpoint SHA256
3f36af30b686f23d0868482675177b29cc50c2b378577dd12f9092ddb0235b2e.
All 484 original state tensors load after value-preserving sparse-kernel layout
conversion. No communication codec is active in this run.

Run003 completed all 3769 original validation frames, seed17, batch1, four
spawn workers. Detector forward receives only stereo images, calibration,
image shape, frame ID and batch size; GT recall/evaluation occur after
prediction. No GT depth, labels or LiDAR reaches prediction forward.

| Metric | Easy | Moderate | Hard |
|---|---:|---:|---:|
| Car 3D AP_R40, IoU0.7 (%) | 86.855334 | 67.723812 | 62.028676 |
| Car BEV AP_R40, IoU0.7 (%) | 92.191437 | 77.444836 | 72.032718 |
| Pedestrian 3D AP_R40, IoU0.5 (%) | 45.496412 | 37.785741 | 32.077540 |
| Cyclist 3D AP_R40, IoU0.5 (%) | 60.007740 | 37.317236 | 34.257933 |

Independent auditor passed: exact frame set/order, 22388 predictions across
classes, 52 empty frames, finite fields, positive dimensions, prediction-pickle
agreement with KITTI text (including h/w/l and output precision), raw label
agreement with validation infos, frozen source/data identities, and fresh
unchanged KITTI AP computation for all three classes (tolerance1e-6).
It does not freshly rehash every image/LiDAR file; archive SHA/CRC identity
comes from the separately completed preparation/download records.

Local evidence: `data/runs/liga-clean-seed17-003/{run.json,metrics.json}`,
`data/runs/liga-clean-audit-003.json`, and
`data/engineering/liga-clean-{seed17-003,audit-003}.log`.
Remote predictions:
`/mnt/d/paper6/checkpoints/liga-author-download.eval/eval/epoch_6/val/clean_seed17_003`.
The legacy directory epoch_6 is parsed from the paper6 path; actual checkpoint
metadata is epoch53. This is not checkpoint selection.

Frozen sources are `*-source-manifest-sensor-spawn.json`, recoverable with
project commit4931244 and its pinned upstream patch. Later training fixes
have separate source manifests. The two zero-prediction startup failures
remain preserved with diagnosis and historical launcher overrides.

This establishes our complete clean detector reference under the recorded
runtime/split/evaluator. It does not reproduce Cao et al.'s communication codec,
prove exact equality to a reported upstream paper table, or establish any gain
from geometry allocation or student compression.
