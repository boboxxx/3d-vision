# True nominal10/50 source training, seed17, isolated execution008

Lock before CPU or GPU execution. Keep the running original30 code003 and the
source architecture005 unchanged. Reuse005 actual global/key factors6/1 and8/4,
all768 states/718 parameter tensors, parameter counts12968818 and13501758.
The old WirelessVariant is used only as a state container for frozen channel
networks; its fixed6/2 radio forward is never called. No radio compatibility,
published author AP reproduction or lightweight encoder claim follows.

The frozen stages.py source-stage2 masks are padded to6, which is incompatible
with the true50 architecture's padding to8 for many source sizes. The new
source_loss.py creates the same checked integer-box native masks and pads to
lcm(global_factor,key_factor). Stage2 uses these padded masks; stages3/4 pass
native masks to005 encode; stage1 global warmup is unchanged. Preserve
Charbonnier epsilon.001, elementwise mean, hybrid masked+.5global, original
phase rates/frozen-flow schedule, configure and optimizer groups. Never change
the existing30 losses, models, source records, thresholds or running jobs.

CPU engineering is precisely synthetic263x265, not full native KITTI training.
Seal both raw synthetic input hashes and boxes, PCG17 float32; test six fixed
phases(1,1),(1,3),(2,1),(3,1),(4,1),(4,6) at30/10/50. Each phase must produce
finite same-size RGB/loss, exactly the configured complete active gradient set,
finite full gradients, and unchanged all model states (no optimizer updates).
The30 path must be bit-identical within the host to the frozen original30
initial states, actual loss/output and full gradient ledgers in all six phases.
Test zero padded ROI outside the native crop and correct factor6/8 dimensions.
Run in both local CPU and Artemis CPU environments; record raw input hashes,
runtime and outputs independently. Do not assume cross-environment bit identity.
No tolerance relaxation, phase removal or input shrinking after observing results.

Formal10/50 training is prepared as a separate train008 entry. The exact source
protocol003 full3712/seed17/batch1/noaugmentation/RGB-only guards and12/10/6/10
epochs remain:44544/37120/22272/37120,141056 updates per rate. Each rate first
needs exactly3 discarded genuine native KITTI GPU stage1 updates, fresh public
SpyNet initialization, actual raw+.0 normal exit, independent saved-state/order
native audit and complete transferred local audit. Gate008 must bind both
complete proofs, their exact input/report/verifier hashes, update counts and
current sacct, and local actual exit0. Fresh formal stage1 never uses engineering
weights. Every later stage requires complete preceding stage native/local
closure and exact last checkpoint; no partial predecessor admission.

Preserve003 Adam and clip10, independent audit_training.py policy/state/moment
and counter logic. New audit008 uses the corresponding actual rate architecture
and verifies nominal rate in every checkpoint. Audit is complete saved-state,
order, finite record and Adam counter/moment evidence, not an independent
recomputation of every CUDA forward/backward or numerical Adam transition.
Save all epoch checkpoints, failures and actual scheduler exits. Source input
locks include all frozen003 sources/metadata,005 architecture and these new008
sources. Formal runtime gates are exclusive actual-result artifacts, generated
only after complete proofs and normal exits. No engineering or formal GPU job
is submitted by the CPU check itself; scheduling is a separate recorded action.

Fixed final stage4 epoch10 is the source checkpoint; no AP selection. Stage5,
actual8-bit quantization/packet charging, reliable ROI control/perfect CSI main
radio contract, calibration and all3769 detector endpoints remain required.
This source implementation does not complete the three rates,83 epochs,
whole radio/resource comparison, main novel-method assessment or manuscript.
