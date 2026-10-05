# KITTI ECSIC model-bound real-byte codec and receiver preparation

Prelock before format tests, final-weight extraction or new entropy encoding.
Preserve all previous public-Cityscapes P6ECv1 sources, containers and receiver
results unchanged. The new P6EKv1 format retains their exact positive finite
CDF, scale bins, residual support/escape encoding, rANS algorithm, four-stream
order and166-byte header/descriptors/CRC overhead. It binds the portable final
model file SHA256, unchanged official config SHA and frozen CDF SHA explicitly.
Do not monkey-patch the old codec's fixed MODEL_SHA. A receiver validates its
expected model against the received header before constructing a neural model.
Wrong model/config/CDF/magic/version, even with correct CRC, must reject.

CPU gate now: reframe the two actual old source containers (synthetic32x64 and
native training000000) with the same public weight identity, preserve every
entropy-stream byte, compare direct independently unpacked header fields and
all four descriptor lengths/positions. Exercise two distinct model identities,
reject cross-model parsing and CRC-consistent tampering, all truncations and
trailing bytes. Compare local and sheng complete fixture ledgers and failures.
These are format/input-admission engineering checks, not KITTI adapted neural
reception, actual candidate calibration or new accuracy evidence.

Prepared final-model registry: only after all six candidate37120-update/10-epoch
runs finish normally and native/local formal003 audits agree on complete
records and checkpoints, extract all225 unmodified final model states to
portable model-only safe PT files. Preserve final-checkpoint SHA, per-state
SHA, lambda/seed/epoch/update count, training report and both audit identities.
Registry admission requires all six distinct predeclared lambdas, not a selected
subset, exact official architecture/config/source identities and the complete
KITTI manifest. Keep the trained216 independent parameters/9 shared aliases.
Receiver loads only a registry-authorized model file with matching SHA, strict
state keys/tensor SHA and finite values; no optimizer or source encoder input.

Prepared neural sender/receiver follows the official frozen evaluation graph:
full native RGB8/255, replicate right/bottom padding32, public positional
encoding; round(x-loc) integer residuals. Streams decode in dependency order
z_left,z_right,y_left,y_right, deriving probabilities only from the selected
model and previously decoded symbols. A fresh byte-only receiver must call
E=0,HE=0,HD=1,D=1, keep all225 states unchanged, have no parameter gradients,
read no clean PNG/GT/calibration/LiDAR/source NPZ, and save complete18 causal
arrays plus full public-dimension top-left crops. No resize/clip/uint8-rounding
of reconstructed float RGB before downstream detection. Formal source-rate
calibration uses the fixed64 training pairs specified in adaptation002, all
six models, complete P6EK containers and fresh receiver proof; estimates never
substitute for serialized bytes. Require an execution addendum and numerical
neural engineering closure before launching this64-pair calibration or AP.

This step prepares input/model identities and codec/receiver implementation.
Training and exact final-model identities remain external prerequisites. Do
not install dependencies/change active pipelines, fit a CDF on validation,
select by AP/quality, narrow original scenarios, or claim completed calibration,
physical radio, whole KITTI validation or new-method benefit from this gate.
