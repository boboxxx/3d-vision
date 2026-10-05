# Received-header digital framing engineering

Protocols36bdef4 and serialization repaire4eb1bc precede actual CPU measurements.
Artemis CPUjob11424046 completed0:0, no GPU TRES.68 actual codec/malformed-frame
checks passed both locally and in the pinned digital runtime; the first local
serialization failure is preserved. All10 actual packet records,18 raw artifacts,
all noise/decoded information blocks and the independent full APP/header/CRC/
padding/resource audit pass. This is engineering, not detector AP or BER curves.

The receiver gets only decoded information blocks and publick. It derives byte
lengths from the20-byte header carried in the SAME LDPC/QAM transmission; all
header bits, whole-code-block padding and realized energy are charged. It never
uses transmitter source length or original RGB to repair a failure. Header,
capacity, padding, CRC and codec failure erase the entire stereo pair.

| Actual packet | 64QAM rate2/3 | 256QAM rate1/2 |
|---|---|---|
| Native000000 JP2rate30, identity | exact90545-byte recovery,181116uses | exact90545-byte recovery,181278uses |
| Fixed8x12 JP2, AWGN6 | erased:header_magic,1063source bit errors | erased:header_magic,1083errors |
| Fixed8x12 JP2, AWGN18 | received,0errors | received,0errors |
| Fixed8x12 JP2, Rayleigh6 | erased:header_magic,1225errors | erased:header_magic,1181errors |
| Fixed8x12 JP2, Rayleigh18 | received,0errors | received,0errors |

Every noisy small64QAM packet consumed1296uses/968.380952energy;256QAM consumed
1215uses/936.082353energy, including all four failures. Actual native energies
180816.571429/181008.776471. Constellation-populationEs1 does not imply exact
per-packetEs1. No ARQ or favorable draw selection. The tiny fixed packet outcomes
are not estimates of KITTI image erasure rates or noisy native-image performance.

The first independent local audit rejected native JPEG2000 pixel hash equality.
It is preserved, not relabeled passed. Exploratory supplement4aff4cb then used
an independent raw-block/header/parser on the original Artemis CPU runtime;
CPUjob11424047 completed0:0, noGPU, allSIX successes reproduced original wire
and image hashes exactly. Complete supplementary pixel arrays were transferred
and verified. Local native re-decode differs in48left/42right values out of
1358640perview, all by1uint8 level. Both code configurations have the same
recovered bytes and identical on-platform pixels. Four synthetic successes
match locally exactly. Pillow versions12.2local/12.3Artemis share OpenJPEG2.5.4;
mechanism of the difference is unresolved. No numerical tolerance was chosen to
pass the original pixel equality assertion. Local audit now expressly proves
bitstream/channel/framing integrity and independent on-platform pixel replay,
while reporting cross-platform differences separately.

Main comparisons must use one fixed audited decoder runtime or identical
serialized received pixel tensors across hosts. Receiver byte-length oracle is
removed for this concrete format. Ideal radio frame boundaries/public code
configuration remain assumptions; real synchronization waveform/control cost,
learned6-bit calibration and ECSIC actual entropy streams remain open. This is
a declared framing/code-family variant because the original settings are unknown.

Evidence: `data/provenance/digital-framing-CPU-001-local-verification.json`,
`data/engineering/artemis-digital-framing-terminal-001.json`,
`data/engineering/artemis-digital-framing-pixel-terminal-001.json` and complete
packet/pixel fixtures under `data/engineering/artemis-digital-framing-*`.
