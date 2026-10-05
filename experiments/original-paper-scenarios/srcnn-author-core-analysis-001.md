# Official SRCNN core recovered and numerically checked

The required comparator now has a pinned primary author implementation and
all eight supplied luminance models. The official project links both recovered
archives: [SRCNN author release](https://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html).
Test archive7558018bytes SHA256
bfa68ca613c1326a59e0c34353205a254ab2b67e34df7f04e28eef567980af30;
training archive19619221bytes SHA256
001146419f7acfb12a3e7929c8acd5de88a08d687d6881085f81321ad6982b1a.
They remain in ignored third_party locally and nativeD storage, with all
model/source hashes in the complete CPU reports.

Protocolae1eeb1 preceded implementation5832e04 and every forward. All eight
MAT models and each of six learned arrays are read and pinned. The mathematical
core uses MATLAB-column-major filter slices, same-size replicate-boundary
correlation, two ReLUs and final linear luminance output, preserving all six
readonly tensors. An independent SciPy implementation interprets original MAT
slices directly and adds channels in the literal author order.

All8models×3fixed synthetic16×20double-input cases pass on both hosts:
24comparisons/7680output values each. Maximum absolute errors are
2.220446049250313e-15 locally and1.7763568394002505e-15 on sheng against their
respective independent references, below the prelocked1e-10 threshold.
Complete source/weight/input identities match across hosts and every tensor
stays unchanged. Cross-host raw outputs are not claimed bitwise identical.
There is no optimization, KITTI, GT, radio, source-quality or detector input.

This is a CPU mathematical translation of SRCNN.m with double input, not an
actual MATLAB execution or released-demo preprocessing/mixed-precision replay.
The test release supplies luminance models at integer2/3/4 upsampling factors
and uses bicubic preprocessing/modcrop/border shaving in its example; its
provided weights do not identify a full native-stereo10/30/50×compressor.

Cao's source comparison names SRCNN,8-bit neural outputs and dimension-based
operating points, but leaves its concrete compression adaptation unresolved.
See [Cao SectionV and reference37](https://arxiv.org/html/2502.12735v1).
A separate explicit compression/color/resizing/quantization/operating-point
and training protocol is still needed. Never present one arbitrary x3checkpoint
as those three published baselines, silently crop calibration coordinates or
call this author-core gate a completed compression/AP experiment.
