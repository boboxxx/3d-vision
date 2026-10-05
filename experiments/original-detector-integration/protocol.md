# Original RGB communication to native detection: integration gates

Lock integration choices before any trained original-codec AP is observed.
This is a downstream protocol for the explicitly declared Cao2025 variant,
not an amendment to its frozen five-stage training/source. Keep integration
in its separately hashed directory while the F6 root executable trees remain
frozen. The original83-epoch cycle and its21 source files stay untouched.

Formal source is only the sole final stage5 epoch45 checkpoint after the
complete predecessor chain and independent stage5 audit pass. Strictly load
all768 states, preserve exact dtype/shape/values and hashes, model/allmodules
eval and torch.no_grad. Record actual final checkpoint SHA when available.
Partial-stage, engineering, best-epoch and new adapted weights are excluded.
Interim engineering may validate interfaces only and cannot supply reported
baseline AP. No original detector GT training or native stage changes here.

## Air and receiver separation

Read paired native RGB[1,3,H,W] in0..1 and independently audited sensor-only
YOLO boxes from the fixed ROI cache. Use the existing original semantic
encoder, exact TableIX9-channel payload construction, serialized noisy control,
CRC and physical channel. Receiver entry point accepts only the observation;
call original receive_payload(observation) then semantic.decode(received).
No private sender boxes/masks/clean RGB/layout/scale or actual fading coefficient
may enter receiver decoding. No reconstruction MSE against clean RGB is needed
in downstream inference. Public detector calibration remains a detector input
under the same assumptions as clean references, not a codec input.

Return physical data/control/pilot/total uses and energy for every attempted
frame, including CRC failures. Erasure means an empty native KITTI prediction
file, never clean fallback, clean mask replay, previous-frame substitution,
retry-until-success or dropping that frame from the AP denominator. Unexpected
nonfinite output, malformed layout or execution errors stop the run; they are
not reclassified as legitimate radio erasure. Measure and report all failures.

Keep reconstructed RGB as finite float32 at native uncropped sensor geometry;
no uint8 round-trip, clipping, JPEG, resize or hidden clean residual before
native detector preprocessing. The original semantic output has no clipping,
so retain out-of-range values and record extrema/fraction. This is an explicit
integration choice because the paper's deployment conversion is unresolved.
A clipped conversion, if later studied, must be a separate declared condition.

## Native downstream receivers and endpoint tests

For author Stereo-RCNN branch1, reverse decodedRGB toBGR, multiply255 and run
unchanged released prep_im_for_blob (including means, resize and native right
crop). The same received images feed dense alignment and the complete original
3D solver. Reuse verified native KITTI writer and branch1author670-state
checkpoint. Do not use clean RGB for alignment or proposal refinement.
Check exact clean relay preprocessing equivalence to original OpenCV BGR path
before any noisy/learned endpoint claim; decoded float range must be preserved.

For LIGA, use its verified author484-state checkpoint and native eval RGB
crop/pad/normalization/calibration pipeline. Disable student and both new links.
Retain sensor-only inference boundary, no feature/GT/LiDAR teacher calls, no
clean feature fallback. Its clean relay equivalence and all484 read-only states
need independent native GPU verification before formal integration. The LIGA
endpoint remains to implement; Stereo-RCNN adapter alone does not finish it.

Final original-codec comparison first uses fixed372 exploratory holdout,
seed17, identity and AWGN10dB, and the same decoded outputs for both detector
endpoints. These are predetermined conditions, no final-main-val tuning or
best noise/channel/checkpoint selection. Preserve each decoded-output identity
and full orderedframe coverage. Audit calibration/GT/predictions, IoU.7 AP_R40
and secondary IoU.5 R11/R40, all fullcodec/detector read-only states and source
identities. Author-pretraining holdout overlap remains explicit.

## Limits and following gates

The trained original9-channel operating point has roughly200065 mean complex
uses with sensor-dependent ROI control versus the current62400 feature point.
Their full native duration/energy/compute and training exposures differ; these
points cannot establish equal-resource superiority. The direct3D method's F1,
F2, F5b andF6 GT exposure requires matched detector/task-adaptation controls,
as do any future uncertainty/risk predictors. Completing this adapter and its
CPU tests is engineering only. Full83-epoch audited weights, full native GPU
endpoints, AP/resource audits and trained matched operating points are still
required for the project's baseline and method claims.
