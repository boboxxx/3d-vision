# Final trained original-codec native detection evaluation

This fixes execution details of the already locked parent integration protocol,
before any trained original-codec AP is observed. It does not change the frozen
five-stage training, the channel, either detector, or the exploratory fold.

Use only the sole stage-5 epoch-45 complete768 checkpoint after all five complete
independent audits pass, the actual original cycle is terminal, and all21 source
files/protocol/current parent checkpoint hashes match. No partial stage, best
checkpoint, new fine-tuning or engineering state is permitted. Preserve the full
12/10/6/10/45 stage chain, actual erasure/Adam budget and all retained audit files.

Encode/decode native RGB on CPU with the unchanged original eval model, sensor
YOLO cache and observation-only adapter. For identity and AWGN10, independently
start a dedicated CPU Generator17 and consume it in the fixed372 frame order.
One radio attempt per frame, no retry. This noise schedule is identical for the
two downstream receivers because the decoded tensors are generated exactly once.

Retain both received float32 CPU tensors losslessly on /mnt/d, with dtype/shape/
array-byte SHA and serialized-file SHA. Store all physical data/control/pilot
uses/energy and true erasures, per-frame sensor hashes and RNG states/digests.
The received-cache manifest is sealed only after full372 coverage, full768
read-only state and source closure. Independent cache audit verifies every
tensor/file hash, raw sensor provenance, receiver scope, CRC-erasure retention,
native geometry, range values and independently recomputed resource formula.

Both native detector processes read the same verified received cache, without
loading the codec or clean RGB for refinement. Public calibration enters only
the detector. An erased frame writes an empty KITTI file and invokes no detector;
all372 frames remain in the AP denominator. Finite out-of-range decoded floats
are preserved, without clipping/uint8. Clean relay references use the same native
sensor cache and fixed372 order and are separately identified as clean references.

Run six predetermined endpoints, seed17: each author detector (Stereo-RCNN670,
LIGA484), each clean relay / identity / AWGN10. No threshold selection (Stereo
fixed .05), no detector adaptation. Native LIGA NCCL/world1/DDP preserves grad
flags but all inference is eval/no_grad. Stereo image backbone and full dense3D
alignment must receive the actual processed received tensors. LIGA image backbone
must receive the actual processed received tensors; all new links/student and
GT/LiDAR teachers/training losses are forbidden. Full detector states read-only.

Freshly audit all KITTI prediction files, calibration/label hashes and empty-frame
coverage after inference. Compute unchanged KITTI IoU.7 Car AP_R40 and secondary
IoU.5 R11/R40 from saved text predictions; independently recompute after sealing.
Paired endpoint audit requires identical received-cache/tensor identities across
both detectors. No inference timing or communication efficiency claim is made.

Require the original cycle PID to be terminal and the pending native queue fully
closed before native final evaluation. Initial physical NVIDIA free >=12GiB;
measured peak+2GiB margin. Preserve unique failed runs without overwriting. Native
single-frame/fresh-768 engineering caches may test implementation only, under a
separate explicit engineering flag/output, and never become a formal initializer.

Results describe the declared reconstruction variant at its sensor-dependent
9-channel operating point, not exact author-code reproduction. Full372 is an
internal exploratory fold with author detector pretraining overlap. The roughly
200k-mean original rate and direct3D62400 rate/training histories are unmatched;
these endpoints alone cannot establish fair rate/energy/exposure superiority,
novelty, mainval generalization, or project completion.
