# ECSIC official checkpoint and actual entropy source bytes

The two fixed engineering inputs now have independently verified real source
containers and exact decode-only reconstructions. This closes a concrete missing
interface needed for the original paper's digital comparator. It does not close
the KITTI-trained comparator, channel evaluation or detector AP.

The public Cityscapes lambda0.01 checkpoint is the README example, not a point
selected on KITTI results. Official source696f4ae4, configba02cc12 and
126636561-byte weighte66b55e2 are fixed. Both z/y context and positional encoding
are enabled. All225 tensors load exactly with weights_only/strict state matching.
Runtime is sheng CPU, torch2.5.0+cu118, two threads, no optimizer or GPU use.

Stage A uses synthetic32x64seed1907 and training000000, with right/bottom
replicate padding only. Native370x1224 becomes384x1248. The independent process
receives integer residuals in zL,zR,yL,yR order and recomputes conditional
location/scale parameters. E/HE calls are forbidden; actual calls are0/0/1/1
for E/HE/HD/D. All18 arrays (symbols, locations, scales, latents, unclipped RGB)
match the unchanged official forward exactly for each input. Complete model
states remain unchanged. Independent saved-array/checkpoint audit passed.

Stage B supplies bytes alone to a fresh receiver. The declared source-codec
adaptation uses a fixed256-level scale grid, 16-bit Laplace CDF, residual
alphabet[-255,255]+escape, byte rANS and a bounded P6EC container. All header,
length, four state-flush, escape and CRC bytes are counted. Table SHA is
507d22e35fb0bb605f05a3312600a8aff3ff10f1ec1c4964358cb390e35d56e5.
This is our explicit finite-CDF implementation, not an author-released coder.

| Fixed engineering input | zL bytes | zR bytes | yL bytes | yR bytes | Header/CRC bytes | Total bytes |
|---|---:|---:|---:|---:|---:|---:|
| Synthetic32x64 | 15 | 20 | 318 | 294 | 166 | 813 |
| KITTI training000000 | 2173 | 2584 | 19968 | 19933 | 166 | 44824 |

The four stream lengths include each rANS flush. Neither native case needs
escape bytes; fixture tests cover escape boundaries/extremes. Native counts are
22464symbols per z stream and359424 per y stream. The byte-only receiver
reconstructs all18 arrays exactly, including independently derived scale-table
indices; all225 states stay fixed. The full native auditor separately parses
the complete container, verifies CRC/lengths/overhead and re-encodes all eight
actual streams with pinned independent upstream C++ rANS; every byte agrees.

Local/sheng CPU fixtures agree with the independent C++ implementation for
131072 alphabet-by-table symbols,10000zeros and8192random varying-table symbols.
All204 malformed-input cases are rejected. Review identified missing interior
scale-boundary coverage; a separate high-precision midpoint test adds all510
transition sides and representable computed-log half ties without changing the
sealed implementation or rerunning native measurements.

First stageB001 attempt failed after writing the synthetic reconstruction:
the NPZ-read barrier rejected its own subsequent output-hash read. Failure,
payload and source remain retained. Prelocked repair04d6a7d hashes an in-memory
NPZ before writing; the read barrier remains active. Actual helper regression
passes locally/sheng. Unique stageB002 then completes both fixed cases. No
equality requirement or coding parameters were changed after native observation.

All transferred metadata and two actual payloads pass local hash/CRC checks.
Large reference/reconstructed NPZ arrays remain on sheng and were fully compared
there. A parallel read-only code review found no blocking correctness issue;
it did not independently rerun those large arrays. Source-byte interoperability
is established only in the fixed CPU runtime, not across GPU/other libraries.

Evidence: data/provenance/ecsic-recovery-stageA-001-audit.json,
ecsic-entropy-stageB-002-audit.json and ecsic-transferred-verification-001.json;
the complete small records/payloads are under data/engineering/ecsic-*stage*.
Protocols and failure repair are adjacent to this analysis. Original sources
and the separate F8 frozen trees were not modified.

Next integration must place complete P6EC bytes inside an explicitly versioned
physical-link header, decode the received LDPC information blocks without the
sender's length, and charge both framing layers and every attempted block.
Crop right/bottom padding using received dimensions before detector preprocessing;
define reconstruction dtype explicitly. Main10/30/50 operating points, training
adaptation, original SNR/fading matrix, detector AP and cross-seed work remain.

Primary references: [official ECSIC code](https://github.com/mwoedlinger/ecsic),
[pinned rANS implementation](https://github.com/rygorous/ryg_rans/blob/c9d162d996fd600315af9ae8eb89d832576cb32d/rans_byte.h).
