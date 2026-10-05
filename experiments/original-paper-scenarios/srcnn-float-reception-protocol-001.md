# SRCNN float reception and shared detector input: prelock

Lock before this receiver implementation, full native reconstruction or SRCNN
AP. This is the declared KITTI adaptation of the unspecified original SRCNN
compressor, not a claim of exact reproduction or a new method. The frozen
training002 model and source interface remain unchanged.

## Byte input, model and numerical operation

The source uses the existing P6SR v1 32-byte header and raw reduced RGB8
payloads, nominal10/30/50, floor(H/sqrt(C)) by floor(W/sqrt(C)). All header and
payload bytes enter the measured ratio. The receiver accepts only the actual
wire bytes and a static six-state model; it obtains native geometry from the
validated header and CRC. No clean image, sender metadata, disparity, GT,
calibration or detector input enters image reconstruction.

Formal weights must be the final20th-epoch,37120-update checkpoint at the same
rate from the complete three-rate formal002 training closure, with all native
audits and complete local training proof. Engineering uses only the terminal
six-update engineering002 checkpoints, separately identified and discarded.
Load checkpoints on CPU with weights_only=True, strict six float32 finite
states and exact final-state hashes. Check rate, epoch, update count and source
identity before loading. No optimizer, gradient, mixed precision, TF32,
clipping or weight update during reception. Set eval and all requires_grad
False. Native CUDA reception uses deterministic cuDNN, benchmark False, no
TF32, seed17 and CPU threads2; no cross-host bitwise equality is assumed.

For each received low RGB8 view, use the existing per-channel Pillow mode-F
BICUBIC interpolation to the header H/W, then the explicit float64 studio
YCbCr matrix and offset from the frozen interface. Cast Y once to float32,
run the trained float32 SRCNN at full native geometry using replicate
boundaries, and replace only Y in the float64 color array. Invert the same
color matrix in float64, then cast the resulting HWC RGB once to contiguous
little-endian float32. Preserve every finite value, including negative and
above-one values, and every native pixel. No author-demo modcrop, shave,
uint8 quantization, RGB clipping or postprocessing selected by AP.

Store left/right HWC float32 arrays in an uncompressed NPZ with exactly those
two keys; this is received-task storage and is not transmitted bandwidth.
Record wire SHA, full array SHA including dtype/shape, cache SHA, all six
read-only parameter hashes, native geometry, model provenance, minima/maxima
and out-of-range fraction. Both downstream detectors consume the identical
NPZ bytes and array hashes. Float adapter makes contiguous BCHW float32
tensors with no divide-by255, rescaling, clipping or geometric change.
Existing native detector preprocessing subsequently uses RGB times255,
Stereo-RCNN's native resize and LIGA's existing bottom crop/pad/calibration.

## Gates and independent audits

CPU engineering first verifies: actual wire parsing and malformed framing/CRC
rejection; float32 SRCNN versus independent SciPy correlation of the loaded
six filters; independent Pillow/color inversion versus every synthetic output;
readonly model and received-only causality; exact float NPZ/tensor identity
and overshoot preservation, plus forbidden clean/GT/foreign-cache reads.
Use all three rates and distinct stereo views. Numerical model comparison
tolerance is2e-5 absolute; this is a tolerance check, not hash equivalence.

After native training002 engineering closure/local proof, run full native
000000/000003 train pairs at all three rates with their discarded engineering
weights. Separate source encoding and receiver processes; install file-read
barriers after static checkpoint/model construction. Require free>=12GiB and
measured peak reserve+2GiB<=initial free, no competing GPU process. Complete
native independent header/CRC/resize/color/pixel/state audit and local artifact
verification precede six native detector engineering endpoints. Engineering
reads no labels and computes no AP. Do not interrupt the main JPEG/JP2 cycle.

Full formal SRCNN training must close before source encoding all3769 ordered
public validation pairs at three rates. Complete received-cache and pixel
audits and local metadata verification precede six full author detector
endpoints. Reuse frozen670-state Stereo-RCNN and484-state LIGA with source-
independent calibration, inference clean/GT/foreign-cache barriers and official
AP after each complete prediction set. Independently replay all native arrays,
prediction/calibration/label identities and official AP, then verify all22614
transferred predictions and six endpoint reports locally. Never rerun a failed
ID or switch checkpoint/resize policy after seeing results.

Quality/ROI measurements require their separate prelock. Right2D remains
unresolved; source-only results do not imply radio delivery, matched training
exposure or complete original-paper matrix. JPEG/JP2 main results, F9 negative
geometry evidence and SRCNN six-step losses were previously inspected; none
selects the receiver operation, checkpoint or these three rates.
