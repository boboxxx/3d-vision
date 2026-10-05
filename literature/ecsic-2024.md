# ECSIC — comparator recovery, 2026-10-04

Matthias Wödlinger, Jan Kotera, Manuel Keglevic, Jan Xu, Robert Sablatnig.
WACV 2024, pp.3436–3445.
[Paper](https://arxiv.org/abs/2307.10284),
[official source](https://github.com/mwoedlinger/ecsic).
Pinned inspected commit: `696f4ae4f250bb1fc750ae9ec23e9c98c2c7e6db`.
Repository downloaded for read-only inspection, no training/dependency scripts executed.

This is a required source comparator in the Cao reference. The official model
uses epipolar cross attention and conditional left-to-right latent/hyperlatent
probability models. The release computes estimated bits from probabilities in
`ecsic/metrics.py:75–93`; `test.py:100–104` averages left/right estimated bpp.
`ecsic/models.py` returns quantized tensors, reconstructed images and rate
estimates. A full Python-source search found no arithmetic/rANS encoder,
decoder or serialized source-byte API in this inspected release.

Thus official rate estimation alone cannot supply LDPC input bits or establish
physical channel-use matching. A real entropy-coder implementation and an
independent decode-only reproduction are required for joint digital comparison.
This is a limitation of this inspected release, not a claim that ECSIC as a
method cannot be entropy-coded.

The README links trained weights on Google Drive and describes Cityscapes and
InStereo2k examples. KITTI-specific weights/settings used by Cao are unknown.
Dependencies pin torch2.1.1/CUDA12 and wandb; do not install these over the live
sheng environment or enable external experiment logging by default. Any
Blackwell deployment requires a separate verified compatible environment.

Recovery update: the public README Cityscapes lambda0.01 config/weights are now
checksum locked and strict-loaded. Two fixed engineering pairs pass independent
four-stream decoding and a declared finite-CDF/rANS source-byte implementation:
813/44824-byte containers, exact18arrays each, all225 states unchanged. Full
native independent C++ byte audit passes. See the experiment's
ecsic-entropy-analysis-001.md for failure retention and evidence. This is not
KITTI adaptation or measured Cao/ECSIC AP. Physical LDPC/QAM integration and
the original operating-point/main-evaluation comparison remain required.
