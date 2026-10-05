# ECSIC real entropy source bytes — fixed engineering contract

Prelock before entropy implementation/measurements. Execution requires successful
stage A closure under ecsic-recovery-protocol-001.md; preserve any failed attempt.
Use exactly that official source, Cityscapes lambda0.01 weights/config, FP32 CPU
runtime, public padding and two fixed input pairs. No new scenes, training,
checkpoint selection, AP or rate tuning. This is a declared finite-CDF coding
adaptation, not recovered Cao KITTI training or the author's released coder.
Do not edit sealed stage A or F8 source trees.

## Probability representation

Four streams in dependency order z_left,z_right,y_left,y_right, contiguous NCHW.
Each symbol is round(latent-location); the decode-only receiver recomputes all
locations/scales from weights and already decoded symbols, as in stage A.
No scales, locations, source features or integer arrays are supplied to it.
Quantize scales (clamped to [0.1,256]) to nearest of 256 log-spaced levels,
using float64 log-index calculation and round-to-even, fixed endpoints. Log
scale quantization is a disclosed implementation approximation, not optimization.

Alphabet has 511 signed residuals [-255,255] in ascending order and one escape.
Mass is zero-mean Laplace probability in [q-0.5,q+0.5]; escape gets both tails
beyond +/-255.5. Normalize nonnegative float64 masses. Integer frequency total
65536, minimum frequency1 per alphabet entry. Allocate floor(p*(65536-512))+1,
then remaining units to largest fractional remainders, ties lower symbol index.
Freeze exact 256x513 little-endian uint32 cumulative-table bytes and SHA before
native measurement. Receiver verifies this table identity. Record the difference
between actual bytes and probability estimates; neither is substituted for the
other. Residuals outside support use escape plus canonical unsigned LEB128 of
zigzag signed integer in source order; absolute residual must be less than2^24.
At most four bytes per escaped integer; reject noncanonical/in-support escapes.

## rANS and framing

Byte rANS, precision16, normalization lower bound2^23, reverse symbol encoding,
reverse emitted bytes, 4-byte little-endian initial decoder state. Require final
decoder state2^23 and no unused/truncated bytes. Integer update equations follow
[Fabian Giesen's public-domain reference](https://github.com/rygorous/ryg_rans/blob/master/rans_byte.h).
Validate independently against a pinned compiled reference fixture (including
nonuniform frequencies, all 512 symbols, varying table choices and renormalization),
not solely against the new implementation's own decoder.

Container base header is big-endian `>4sBB4H32s32s32s`: magic P6EC,version1,
stream_count4, original H/W,padded H/W, weight SHA256,config SHA256,CDF SHA256.
Four `>BIII` descriptors follow: ordered stream id0..3, symbol count, rANS byte
count, escape byte count. Body concatenates each rANS then escape byte sequence.
Append big-endian uint32 CRC32 over header+body. Fixed overhead166bytes, including
CRC, plus all four flushed states and escapes. Whole container maximum32MiB;
dimensions limited to2048x4096 and valid positive multiple32 padding with less
than32 added pixels. Expected symbol counts derive from dimensions/48channels.
Validate complete header/lengths/CRC before model allocation/decoding. No silent
truncation, guessed dimensions, retry or fallback. Any malformed container erases
the pair. CRC is error detection, not a cryptographic guarantee.

## Validation and scope

First CPU fixtures on local and sheng: exact round trip across all alphabet
symbols/tables, zeros, escape boundaries/extremes, dedicated PCG64seed1908 random
fixtures, scale-index boundaries, empty/malformed streams, wrong order/dimensions,
truncation/append/bit flip/CRC/header identity. C++ reference byte identity and
independent receiver must both pass. Then encode the two stage A fixtures once,
seal actual bytes, and decode in a fresh process. The process accepts only the
container plus fixed public model/config/CDF. Deny source/reference/NPZ fixture
reads and E/HE execution. Compare every decoded symbol/location/scale/latent and
unclipped RGB exactly against sealed stage A arrays in the external auditor.
Audit complete unchanged weights/states/sources, file hashes and actual lengths.
Do not relax equality or tune CDF/support after observing native output.

This closes actual source-byte availability only. LDPC/QAM framing, channel
erasures, reconstruction dtype/cropping for detectors, KITTI adaptation and
main154condition AP remain separate work. Count this complete P6EC container
as source payload when later embedded in the already audited P6SB physical-link
header; both layers' overhead and all attempted physical blocks must be charged.
