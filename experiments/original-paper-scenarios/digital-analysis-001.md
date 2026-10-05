# Actual digital engineering: result and limits

Prelock83e103c precedes pinned source/runtime18f01f4 and transceiverde2aaeb.
Artemis CPU jobs11423924/11423929 completed0:0, no GPU TRES. Independent PHY-only
runtime torch2.9.1+cpu/numpy2.2.6/Sionna2.1.0; sionna-rt deliberately omitted
and pip check's sole missing-RT message retained. Installed PHY Python/CSV bytes
exactly match official pinned revision6498239a72267ee25edb600e5b826b0531971d7f.
Existing original/F7/CUDA runtime sources were not modified.

All12 checks passed: complete Gray constellations/APP independent likelihoods,
full mother-codeword zero syndromes/systematic fillers/explicit RV0 output
interleaving, actual byte encode/QAM/APP/BP roundtrip, exact independent noise
and fading draw replay, and full native JPEG2000 paired-stream recovery. Two
real native370x1224 views from training000000, original90,545-byte P6SB stream,
passed complete byte equality and CRC/native image decode under identity.
Independent local audit checks all8 packet records and16 retained artifacts,
all coded/mapped/received/LLR/decoded arrays, actual resource counts and recorded
bit/block errors. Cross-platform division/APP tolerances and measured roundoff
are explicit; actual saved array SHA and RNG draws checked exactly.

| Configuration | Packet | Channel | Source bit errors | Blocks with errors | Complex uses | Actual payload energy |
|---|---|---|---:|---:|---:|---:|
| LDPC2/3 +64QAM | 262 bytes | identity | 0 | 0/2 | 648 | 505.523810 |
| LDPC2/3 +64QAM | 262 bytes | AWGN10 | 375 | 2/2 | 648 | 505.523810 |
| LDPC2/3 +64QAM | 262 bytes | iid Rayleigh10/ZF | 504 | 2/2 | 648 | 505.523810 |
| LDPC1/2 +256QAM | 262 bytes | identity | 0 | 0/3 | 729 | 668.152941 |
| LDPC1/2 +256QAM | 262 bytes | AWGN10 | 367 | 3/3 | 729 | 668.152941 |
| LDPC1/2 +256QAM | 262 bytes | iid Rayleigh10/ZF | 466 | 3/3 | 729 | 668.152941 |
| LDPC2/3 +64QAM | native JP2-rate30 | identity | 0 | 0/559 | 181116 | 180816.571429 |
| LDPC1/2 +256QAM | native JP2-rate30 | identity | 0 | 0/746 | 181278 | 181008.776471 |

These are fixed tiny-packet engineering checks, not estimated BER curves or
noise10 native image/detection results. Failed noisy packets are preserved;
there is no clean fallback or successful-packet selection. Both code variants
have nominal4 source bits/use, but actual whole-block padding differs, and
constellation-normalized population Es1 does not equal every packet's actual
energy. Hence formula-based uses/energy equality would be incorrect. Packet
length is public engineering metadata; main framing/control accounting must be
fixed before actual main digital comparisons.

Code family/n1944/20-iteration BP/Gray mapping/NR interleaving are explicit
ambiguity resolutions, not recovered original author settings. Main learned
6-bit calibration, actual ECSIC entropy streams, multi-SNR/fading native source
packets, all3769 predictions and matched energy/exposure comparisons remain.
