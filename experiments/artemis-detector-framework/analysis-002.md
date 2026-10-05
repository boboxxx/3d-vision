# Complete author GPU forward and actual detection-loss input backward

Artemis RTX PRO6000 sm120 executes the complete author484-state detector on
fixed training frames000000/000003, sensor-only entry. Forward11424837 and
supervised backward11424856 both actually finish COMPLETED0:0. All author
states/source/runtime remain unchanged; public checkpoint is safely parsed
with weights_only=True. Sparse teacher states load, but its GPU forward is
never invoked. Peak reserved memory: forward4.546875GiB, backward11.595GiB.

Forward produces1/3 post-NMS boxes and saves every native depth logit, raw
prediction and input. Both full NPZs and93,383,301-byte checkpoint now exist
locally. Native closure covers246,928,220 saved values. Local audit003 verifies
all484 states, every transferred array/source hash/shape/finite value, original
RGB normalization/crop/calibration and every post-NMS row in the raw head.
The misleading depth_logits_low field actually has shape[1,1,288,320,1248].
Large NPZ/checkpoint bytes are retained on both hosts and locally, with Git
tracking complete metadata/SHA; exact paths are excluded from Git duplication.

Original strict FP64 depth-mean tolerance3e-5 fails: maximum discrepancies
4.33294e-5 and8.13361e-5. Blank trailing calibration-line parser failure and
both verifier logs are retained. Audit003 records full coverage without
changing tolerance or claiming passed numerical equivalence. A separate
prelocked exact same-runtime FP32 softmax/reduction replay11424868 is submitted;
actualCOMPLETED0:0, all798720depth outputs bit equal to saved native values.
Native and complete local replay proofs both pass. This establishes
exact native arithmetic reproduction, while original FP64 tolerance remains
failed. Precision accounts for the difference to the independently computed
FP64 equation; no native arrays or acceptance thresholds were changed.

Backward freezes all parameters, keeps eval and marks both preprocessed RGB
inputs as gradient leaves. Labels are read after sensor-only forward for
loss-side targets only. Native classification/box/IoU/direction losses yield
total0.0724609569/0.0980025604, with2/22 positive anchors. Both images have
1,198,080 finite nonzero gradient values per frame. All parameter gradients
remain absent and all484 state hashes unchanged. Native closure and independent
local verifier002 cover20,091,058 saved values including4,792,320 gradients;
independent camera-to-pseudo-LiDAR GT conversion matches exactly.

The extra verifier001 post-NMS bit comparison against no_grad forward fails:
boxes differ at most3.06368e-5/7.17640e-5 and scores8.22544e-6/2.07126e-5;
classes/counts match. This bit criterion was not in the backward protocol;
retain the failure, report deltas, and verify original autograd requirements.
No claim of a numerical gradient oracle or output numerical equivalence.
These gates establish actual task-loss access for subsequent codec work,
without implying trained communication, geometry-specific AP benefit,
full main validation, sparse GPU teacher training or submission readiness.

Evidence: [complete forward identity audit](/Users/chen/Documents/ChatGPT/paper6/data/provenance/artemis-full-detector-GPU-local-audit-003.json),
[actual backward terminal](/Users/chen/Documents/ChatGPT/paper6/data/engineering/artemis-detector-gradient-GPU-001/terminal.json),
[all-gradient and independent GT proof](/Users/chen/Documents/ChatGPT/paper6/data/provenance/artemis-detector-gradient-local-verification-002.json),
[locked backward protocol](/Users/chen/Documents/ChatGPT/paper6/experiments/artemis-detector-framework/gradient-protocol-001.md).
