# Explicit wireless reproduction variant

This implements the TableIX example (3→64, three64→64 BN/ReLU layers,
64→9; decoder9→64, five64→64 BN/ReLU layers,64→3). Global and key codecs
have independent parameters; each is shared across views. With the complete
semantic/flow network the variant contains12,970,798 parameters in718 learned
tensors. These choices and the following air format are documented variants;
the original paper does not supply this exact protection scheme or sharing.

Global codes are dense. Key codes are gathered at cells selected by2×2 max
projection of the original sensor-generated box union mask; no guard band is
added. An odd-coordinate box can touch extra boundary cells. Received code
canvases are zero outside those cells. Both key CNNs still compute densely;
sparse transport does not imply sparse computation.

The concatenated global-left/global-right/key-left/key-right real streams are
paired into real/imaginary components, with one zero real padding value when
needed. Transmitter normalization gives total analog-data energy equal to its
complex symbol count. The normalization factor is never provided to the
receiver. This is analog JSCC, with no claimed quantized bitstream rate.

Control uses a13-byte network-order prefix: magic4, version1, native height2,
width2 and box counts2+2. Each box uses four uint16 native coordinates and one
float32 confidence (12 bytes); classes are unnecessary for the fixed union
mask rule. Prefix and body each have a separate CRC32. Every control bit is
sent as seven repeated unit-energy real BPSK symbols. The prefix costs952
complex uses; total control costs1176+672×(number of boxes) uses. Empty ROI
still sends shape/counts and both CRCs. Integer coordinates and float32
confidence have no additional uncharged side information.

AWGN has complex noise power10^(-SNR/10), half on each real component. Block
Rayleigh adds eight explicit unit pilots sharing the frame fading coefficient.
The receiver estimates fading from noisy pilots only and applies LMMSE
equalization. Pilot, control, analog-data and odd-padding costs are all counted.
No true fading coefficient is available to receive_payload. Public channel kind,
noise power and fixed architecture are preprovisioned.

The receiver reads the fixed prefix before trusting lengths, reconstructs masks
from decoded boxes, then validates the observed packet duration against the
decoded layout. CRC, malformed control, missing symbols or degenerate pilot
estimates cause FrameErasure. There is no clean-box or clean-mask fallback.
Erasure handling in training/evaluation must be explicitly counted; evaluation
must emit an empty prediction for the externally scheduled frame identifier.

Channel-only stage5 uses MSE between frozen compressed semantic-encoder
features and received channel-decoder features, averaging global-view MSE and
nonempty key-view support MSE equally. Empty key supports contribute no MSE
term. This reduction and sparse boundary behavior are variant choices; the
paper specifies MSE but not this reduction. The later joint stage uses RGB
Charbonnier. No image target or clean semantic feature is passed to the
operational receiver; targets are used by the training loss only.

The synthetic CPU probe checks full forward/backward gradients, wire control
round trip, a corrected one-copy error, rejection of CRC corruption in either
prefix/body, odd padding, exact data/control/pilot counts, energy, empirical
AWGN variance and an analytic pilot-only fading fixture. It is not a trained
wireless quality or detection experiment. The full-fold budget auditor derives
layout cost independently from the audited ROI records; it does not substitute
for a complete trained-frame channel/AP audit.
