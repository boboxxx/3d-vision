# Training-only JPEG operating points for the original source scenarios

Prelock before measuring new JPEG rates. The original scenario contract requires
nominal raw-RGB8/source compression ratios 10/30/50; JPEG quality is not a
compression ratio. This phase fixes three global qualities by actual serialized
bytes on training images. It does not execute a main-validation condition,
detector, wireless draw, SSIM measurement or per-frame quality selection.

Use exactly the lexicographically first 64 IDs of the sealed internal
geocomm_tune_train fold (3340 IDs, disjoint from its 372 holdout and the main
3769 validation split). Native left/right RGB8 PNGs, full dimensions, no resize
or augmentation. Hash all 128 input PNG files before and after. No GT, calibration,
ROI, main-validation image, checkpoint or model reads are permitted.

On sheng, freeze Python3.10.15, NumPy1.26.3, Pillow10.2.0, libjpeg6.2,
libjpeg-turbo3.0.1 and OpenJPEG2.5.0. Reuse the unchanged previously checked
image_codecs.py and contracts.py. JPEG: every integer quality1…95, 4:2:0,
optimize=false, progressive=false. Encode both images, retain exact resulting
codestream lengths, received geometry and fingerprints, complete P6SBv1 bytes
including its20-byte header/CRC, source PNG hashes and encoder identity.
No arithmetic substitute for real serialization. Fixed64×95=6080 pair encodes,
ordered quality first, frame ID second. Complete decoded geometry must match.

For each quality q, compute R(q)=sum(raw_RGB8_bytes)/sum(full_P6SB_bytes) over
the same64 pairs. Choose q for target t∈{10,30,50} by lexicographic minimum
(abs(log(R(q)/t)), q); a tie therefore selects the lower quality. Report the
measured pooled ratio and residual, including unreachable targets or repeated
selected qualities. Do not silently claim exact target matching, retune the
search, interpolate a quality or choose by reconstruction/detection quality.
One selected global q per target must be frozen before any compressed-mainval
observation. JPEG2000's already declared single rates10/30/50 remain separate.

Large calibration records live on /mnt/d/paper6; small manifest/logs/provenance
may live in the checkout. CPU-only, no shared dependency changes and no edits
to the active eight F9 source trees. Reserve one unique calibration ID before
encoding, preserve any failure. A separate independent audit must check all6080
raw records, exact quality/frame coverage/order, all128 source identities,
byte accounting, environment/source identity, pooled ratios and all three
selection decisions. Records retain byte/hash evidence; this phase does not
retain/replay every candidate wire on a second platform or claim local native
PNG replay. Actual PID terminal and transferred complete records close it.

Full3769 JPEG/JPEG2000 shared received-image caches and their detector inference,
source-quality/ROI/SSIM metrics and original digital SNR matrix require a later
committed execution addendum. This calibration is baseline preparation, not a
novel method or main accuracy result.
