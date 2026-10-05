# Declared SRCNN stereo compression interface: engineering lock

This addendum defines missing implementation choices before observing any
SRCNN compression, quality or detection result. Cao Section V names SRCNN
and 8-bit outputs at dimension-adjusted 10/30/50 ratios, but does not release
this adapter. Label the resulting system **declared SRCNN adaptation**, never
an exact recovery of the paper's unpublished comparator.

Primary evidence: [Cao Section V](https://arxiv.org/html/2502.12735v1),
[SRCNN author release](https://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html), and
[MathWorks studio-range color convention](https://www.mathworks.com/help/images/ref/rgb2ycbcr.html).
The unchanged official core, eight-model numerical checks and source identities
are already closed under srcnn-author-core-protocol-001. Preserve that code.

## Fixed source and received-image interface

Input is a pair of equal native HWC RGB8 images. For nominal C in 10/30/50,
choose low height floor(H/sqrt(C)) and width floor(W/sqrt(C)), each positive.
Pillow 10.2.0 BICUBIC RGB8 resizing supplies the actual 8-bit low-resolution
representation. No image-dependent rate search, JPEG/entropy coding, crop,
calibration, GT, detector or ROI input. Report every actual byte and measured
raw-to-wire ratio; nominal C is not an exact physical compression guarantee.

Serialize a 32-byte big-endian `>4sBBBBIIHHIII` header: magic P6SR, version1,
algorithm1, nominal C, reserved0, native H/W, low H/W, left/right payload
lengths and CRC32 of both payloads. The two payloads are contiguous HWC RGB8
bytes, in left/right order. Reject unknown versions/algorithms/rates, zero or
inconsistent dimensions/lengths, trailing bytes and CRC errors. Dimensions
derive only from this received header; no external shape or clean image enters
the receiver. Static decoder weights are shared system configuration.

The receiver converts the low RGB8 channels to float32 values divided by255,
then separately resizes each Pillow mode-F channel to the header's exact native
H/W using BICUBIC. Record this implementation explicitly: it is not MATLAB
imresize or the author's modcrop/shave example, and it keeps the native stereo
calibration coordinates. No clipping or uint8 quantization after interpolation.

Use the explicit studio-range RGB-to-YCbCr convention, RGB in units0..1:

    [Y,Cb,Cr] = ([R,G,B] @ A.T + [16,128,128]) / 255
    A = [[65.481,128.553,24.966],
         [-37.797,-74.203,112.000],
         [112.000,-93.786,-18.214]]

Compute color operations in float64 and invert this exact specified matrix
to recover RGB. Apply the unchanged CPU float64 author core only to Y; retain
the bicubic Cb/Cr. Reconstructed RGB remains float64, finite, native HWC,
unclipped. Report range and out-of-range fraction. Subsequent model/metric
float32 conversion or clipping needs its own explicit execution policy.

For engineering use only the release's default demo model
`9-5-5(ImageNet)/x3.mat`, fixed before any result. This is a mathematical
pipeline test, not a claim that x3 weights solve all three compression rates.
No model selection across the eight checked weights. Training-only KITTI
adaptation, fixed training budget/optimizer/final weight identities and main
source caches/AP require a separate protocol before their first execution.
There is no radio in this source interface; PHY uses/energy remain undefined.

## Engineering now allowed

Pure CPU checks on local and sheng: independent header/CRC/length parsing and
all actual payload bytes; fixed dimension arithmetic including tiny/invalid
inputs; color primary/black/white reference values and inverse roundtrip;
fixed synthetic32x48 RGB8 views through all three rates, native output shape,
finite unclipped output, same received stream replay, altered received payload
causally changing outputs, and all six fixed author tensors unchanged.
An identity luminance core tests the complete interpolation/color composition
independently of SRCNN filtering. Keep all source/core/archive/model identities
and runtime records; unique outputs only. No KITTI, PNG, GT, calibration,
training, quality score or detector access. Passing this gate does not complete
SRCNN baseline recovery or the original main matrix.
