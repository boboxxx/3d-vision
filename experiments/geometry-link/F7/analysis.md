# Matched channel adaptation, seed17-001

Both prelocked arms completed 3,340 updates from the identical full F6b parent, with fresh AdamW, identical augmented sensor/GT/calibration fingerprints at every step and only 16 codec parameter tensors active. All 519 frozen states, full 535-state snapshots and actual optimizer moments/counts passed independent server audits. The four prescribed 372-frame endpoints passed AP, feature, received-only chronology and read-only state audits. The cycle and wrapper exited; closure rehashed all native prediction/GT/calibration files and frozen source trees. Local verification checks all 46 retained artifacts and both complete raw training streams.

Car 3D AP_R40, IoU0.7, percent:

| Training arm | Identity test, Easy / Moderate / Hard | AWGN10 test, Easy / Moderate / Hard |
|---|---|---|
| Identity continuation | 46.8630 / 25.6075 / 21.9926 | 35.5365 / 17.4029 / 15.0650 |
| AWGN U(0,20) adaptation | 48.5278 / 26.1915 / 21.7488 | 45.7766 / 24.5328 / 20.0046 |

The prelocked primary Moderate AWGN10 treatment minus control is **+7.1299191429 percentage points**. Both endpoint conditions attempt 62,400 complex symbols per frame with actual mean energy 1 and zero header/pilot symbols in this declared uniform feature-transport variant. Both arms include all 62 legal empty-target samples. Model inference has no teacher or clean feature fallback.

This single-seed internal experiment provides controlled evidence that channel exposure during the added task adaptation improves AWGN10 robustness relative to an equal-update identity continuation. It does not establish geometry-dependent allocation, calibrated uncertainty, fading robustness, statistical significance, cross-seed superiority, main-validation results or equivalence to the original paper's detector/rate/SNR settings. The released author detector was pretrained on the original train split containing this internal holdout. Native non-bitwise repeatability documented in earlier runs remains relevant; no confidence interval is invented from four endpoint values.

The previous F6b checkpoint remains the sole parent of the separately prelocked geometry-risk pilot. These results do not silently replace that parent. A future method comparison should include a channel-adapted equal-budget control and explicitly charge adaptation supervision/updates. The main original-aligned AWGN6–18/Rayleigh/perfect-CSI/Stereo-RCNN/IoU0.5 experiment is still required.

Evidence: `data/provenance/stereo-channel-native-seed17-001-closure.json`, `...-actual-terminal-001.json`, `...-local-verification-001.json`; both raw 3,340-row streams and all four AP/feature/read-only audits are retained in `data/runs`. Large weights remain on sheng `/mnt/d/paper6/runs`.
