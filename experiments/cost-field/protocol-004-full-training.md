# Full-split task training before validation performance

Prelock after full32-update native/local closure003. No engineering AP or
loss ranking selects architecture, decoder or hyperparameters.

KITTI train3712 SHA b6417a1d9b18c8fdb085128e633d28ff321b7674a6d1b3841b8f43d865b281cb;
val3769 SHA657ac4bcc1e156e5b106a4ca18e1f88e012787ea1d2b5d0adeea97fee903fa86.
Verify disjoint IDs and every raw stereo/calibration/label byte in an immutable
native manifest before launch. Keep author484 states frozen/eval. Its upstream
checkpoint is not fitted here on validation. Preserve all old sources/runs.

Train G/P/S/B independently exactly3 epochs, batch1, native deterministic crop,
no extra augmentation, seeds17/23/41. Initialize new G/P/S identically within
each seed; initialize B from that seed with identical appearance weights.
Do not reuse optimized003 weights. B has1751 functional parameters vs1650,
not equal capacity. Same19complex symbols/energy19 per pooled site, stride4;
record actual frame shapes/resources. No AP-based initialization tuning.

Adam lr1e-3, betas(.9,.999), eps1e-8, no decay, norm clip10, FP32; real native
cls+box/IoU/direction loss only. No image, depth, teacher or reconstruction loss.
Identical seed-dependent shuffled3712 IDs per epoch for every arm. Identical
step schedule: AWGN/Rayleigh equiprobable, SNR uniform6..18dB, paired noise
and fading RNG by seed/step. No sender instantaneous fading oracle. Keep
perfect-CSI ZF raw physics unclipped; constrain only decoder-domain sphere.
Save noise, fading and received pre-ZF baseband alongside raw tx/rx, so the
whole physical arithmetic can be independently verified.

Keep empty/range-empty GT and zero-positive frames; native assigner accepts
an empty GT dimension, with no artificial zero-class box. Native regression
supports zero positives. Require finite loss/gradients, record zero gradients
honestly, and never discard frames due to their gradient. Check all author/
source states at start/end/each epoch. Save every update's raw arrays, gradient,
optimizer checkpoint, SHA, source references and loss in append-only JSONL.
Do not rewrite an expanding all-record JSON after every iteration.

11136 updates/arm/seed,44544/seed,133632 total. Preserve failed/interrupted
prefixes, never silently restart or reduce epochs. Exact checkpoint/RNG/step
identity is required for any subdivided job/resume. Initially dispatch seed17;
dispatch23/41 after actual early full-data progress is verified.

Evaluate final epoch3 weights only on all3769 IDs. Primary: Car3D R40 Moderate,
IoU0.7, AWGN10, P-B difference per seed and mean. Report every G/P/S/B, easy/
moderate/hard and BEV, all raw predictions and official evaluator version.
P-S is spatial organization, P-G pretrained-prior ablation. Never substitute
a favorable arm/channel for the primary contrast. Three-seed variation is
descriptive, not a population significance claim; image bootstrap is not a
substitute for training-seed variation.

After the primary, evaluate frozen final weights over AWGN/Rayleigh at
6,8,10,12,14,16,18dB and identity. Original RGB/digital/SRCNN/ROI/ECSIC matrix
and matched rate curves remain required, not closed by stride4 training.
Report full sender cost-volume compute/storage: lightweight transmission
and detector portability are unproven.

Large raw records stay on Artemis, with complete bounded-memory local
streaming verification, never sampling. Retain final/epoch checkpoints locally
plus all hashes/metadata/proofs. No positive geometry or submission-readiness
claim before full validation and original-method comparisons.
