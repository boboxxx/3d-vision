# F0 failure diagnosis D0 (locked before diagnostic inference)

F0 completed3340 updates. Checkpoint audit passes all484 original tensors:
481 unchanged (including every original parameter and BN state), with only
global_step and two training-only imitation normalization scales permitted to
update. All63 new parameter tensors have nonzero optimizer second moments;
checkpoint SHA640d326d95d51c0673de00a81ef78c5921903bcf8cf7b40e614fb3c02e9870e0.
Full saved scalar audit passes3340 steps, all finite, energy/symbol equality.
First/last100 loss median13.6494/9.6612.

Native372-frame10dB holdout Car moderate3D AP_R40 IoU0.7=0.0076202% before
independent AP recomputation. The channel-file coverage audit rejected this
auto-evaluation: zero accounting records, despite successful train records.
DDP rebuilds the outer input dictionary; the actual channel measurements are
in the returned sensor batch. Fix logging to use returned prediction metadata,
preserve old results, reevaluate the same checkpoint with explicit seed17 and
a new run identity. The new inference noise stream is different from the
original post-training RNG stream; do not overwrite or demand equal AP.

Pause F0b allocation comparison until an encoder/codec produces useful 3D
features. This changes the next exploratory action in response to a negative
result, not a post-hoc confirmatory protocol change. Do not claim any gain.

D0 uses the same372 codec holdout and frozen complete receiver:
1. Clean released detector, no communication/student, author checkpoint.
2. Trained student with communication disabled, F0 checkpoint, strict required
   student/original tensor loading. This is explicitly uncompressed diagnostic
   inference; no communication efficiency claim.
3. Same trained student+uniform codec at10dB, F0 checkpoint, corrected accounting,
   independent AP/artifact/channel audit.

No learning updates or main-validation tuning. Record sources/config/weights,
all372 predictions, native AP and recomputation. These contrasts separate sender
feature usability from communication degradation. If student alone fails,
prioritize feature pretraining/scale diagnostics; if student alone is usable
but its codec fails, prioritize receiver feature reconstruction warm-up and
pooling/bandwidth diagnostics. Retain all negative results. A longer training
schedule and all matched comparisons must be separately locked before launch.
