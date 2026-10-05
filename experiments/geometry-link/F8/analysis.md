# Matched student encoder adaptation, seed17-001

The prelocked comparison is complete. Both arms finish3340 actual updates from
the identical full F7 AWGN parent; same augmented inputs, empty-target cases,
channel SNR and actual isolated noise draws at every step. All training/state/
Adam audits and four372-frame AP/received-feature/read-only audits pass. The
cycle actually exited; closure rehashed all six source roots, original21files,
native prediction/GT/calibration files, checkpoints and raw records. Local
verification passes all46 retained artifacts and both complete3340-row streams.

Car3D AP_R40 at IoU0.7, percent:

| Optimization scope | Identity Easy / Moderate / Hard | AWGN10 Easy / Moderate / Hard |
|---|---|---|
| Codec only,16 tensors |47.0785 /25.6649 /21.2528|41.9243 /22.3785 /18.6336|
| Student encoder + codec,51 tensors |48.9123 /28.0136 /23.0627|48.1643 /25.6649 /21.7088|

The prelocked primary AWGN10 Moderate difference is **+3.2863695330 percentage
points**. Identity Moderate difference is+2.3487707850points (secondary). This
supports useful encoder adaptation relative to equal-update codec continuation
in this fixed internal experiment. The unchanged inference architecture and
resource budget isolate the chosen optimization scope; this is a stronger
uniform control, not a geometry-allocation contribution or a new architecture.

The parent F7 AWGN endpoint was24.5328% Moderate. Descriptively, the additional
codec-only epoch ends below that parent and the joint epoch above it. The parent
has a different update budget, so the controlled result is the paired F8
comparison. Extra optimization does not uniformly improve performance; no
learning-rate or checkpoint search is justified as an explanation by these
four measurements. No intermediate checkpoint is selected.

Both arms retain62empty-GT samples. Control updates16tensors/519states fixed;
joint updates51/484original detector states fixed, all535 full states initialized
exactly. AdamW is fresh in both arms, learning rate/weight decay1e-4, global
gradient clipping10, native classification/3D box/direction/IoU losses, no
teacher/depth/2D/reconstruction objective. Every sample attempts62400complex
uses with actual mean energy1; fixed49920stereo+12480appearance, header/pilot0
in this uniform feature-transport variant. No clean feature bypass.

Joint training has extra backward computation, not a matched training FLOP
claim. Peak allocated/reserved GiB are5.1930/6.8262(control) and5.7916/7.3340(joint).
Final checkpoints alone are fixed: control e2b76e2b84b0f42a13e9d291753f5031f3c1b4304f3ac147f6fce74e9b008961;
joint6e54ebc5fa696db26b3902cdd129ff42544635db4cfb56dfdb7f4b94841c5193.

The372internal holdout is contained in the released author's detector training
split. This single-seed result does not establish main-validation generalization,
cross-seed significance, calibrated geometric uncertainty, resource allocation,
fading robustness or superiority to the original RGB method at its different
rate/training exposure. Original-aligned main evaluation remains required.

Evidence: data/provenance/stereo-encoder-native-seed17-001-{closure,actual-terminal-001,
local-verification-001}.json, all training/endpoint artifacts in data/runs and
logs. Actual terminal evidence timestamp2026-10-04T17:49:42.365713+00:00. Large
checkpoints remain onsheng. Source freeze is released after complete closure;
sealed F8 code and measurements must remain unchanged for later comparisons.
