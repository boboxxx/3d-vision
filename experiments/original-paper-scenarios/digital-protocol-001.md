# Actual digital transceiver engineering 001

This lock precedes implementation and all encoded/decoded measurements. It
extends the original-paper scenario contract with an explicit digital-code
ambiguity-resolution variant. No mainval, learned model, detector AP, tuning,
or changes to frozen original/F7/root/final receiver sources are allowed here.
Original paper gives rates and modulations but not code family, matrices,
block length, bit labeling, decoder iteration count or bitstream framing.
Consequently this implementation will be reported as a declared variant,
never as recovered original author configuration.

Use unmodified NVIDIA Sionna v2.1.0, Git revision
6498239a72267ee25edb600e5b826b0531971d7f, Apache2.0. Inspect/pin all installed
source/CSV identities; runtime is separate from both active project runtimes.
Its package requires Python>=3.11, torch>=2.9.1, numpy>=2.2.6. Create an
independent Artemis CPU-only environment digital-cpu-001 (no RT module),
with torch2.9.1+cpu, numpy2.2.6, scipy1.15.3, matplotlib3.10.8, h5py3.15.1,
importlib-resources6.5.2 and Sionna2.1.0. Installation reports record public
wheel URLs/hashes; do not modify torch-cu128-001 or sheng runtime. Use a
short CPU job4CPU/16GiB/1h for setup and the later checks. GPU not required.
No existing environment overwrite; failures preserved under unique IDs.

Encoder LDPC5GEncoder follows3GPP38.212, RV0 rate matching, automatic
basegraph/lifting; record selected basegraph/lifting/full PCM identity,
shortening/filler/puncturing and output interleaver. n1944:
- k1296, rate2/3,64QAM (6 bits/symbol), configured NR output interleaver6.
- k972, rate1/2,256QAM (8 bits/symbol), configured NR output interleaver8.
Decode with linked LDPC5GDecoder:20 iterations, flooding, boxplus-phi,
LLRclip20, hard information bits, prunePCM=true, precisiondouble, CPU.
Decoder receives actual APP log p(bit1)/p(bit0), not hard decisions.
No ARQ, no best noise draw, no clean source fallback, no LDPC-outage shortcut.

Actual source bytes (including framing/CRC) unpack MSBfirst, zero pad to
ceil(source_bits/k) whole blocks. Exactlyn transmitted bits/block; both n
are divisible by modulation bits, so no QAM-tail padding. Serialize/recover
only original byte count and validate padding; source byte length is public
engineering metadata, not a claim of free signaling. Any main deployment
must provide a counted length/framing protocol before execution. Existing
P6SB streams are already framed and integrity checked by their actual decoder.

Use Sionna's fixed Gray QAM points, normalized over the entire constellation
(population mean Es1). Do NOT normalize each source-dependent codeword/frame:
that would require signaling its private normalization scale. Record actual
sum|x|² and population Es separately; realized energy is not automatically
number of uses. Nominal SNR uses complex N0=10^(-SNR/10), not Eb/N0, not
received instantaneous power. This digital convention is explicit and differs
from exact per-frame normalized learned analog source; physical-energy-matched
claims require actual charged energies, not nominal equality.

Noise/fading: independent dedicated NumPy PCG64 seeds1901/1902; double complex.
AWGN n~CN(0,N0). Rayleigh iid per symbol h~CN(0,1), perfect CSI,
y=h*x+n and exactZF z=y/h; demapper uses actual noise varianceN0/|h|²
per symbol. No deep-fade floor/clipping, no pilot/CSI estimation. Identity
consumes neither RNG. APP demapping uses logsumexp over explicit constellation
points and bit labels, making the sign/variance independently checkable.
Channel implementation retains noise/h/rng states and SHA for actual draws.

Engineering checks, CPU only:
- Full64/256-point Gray labeling bijection, population energy1, finite APP
  sign and independent direct likelihood comparisons at known points.
- For both(n,k), random/zero/one source blocks: independently multiply the
  full lifted PCM by full mother codeword in GF2 before rate matching; zero
  syndrome and exact systematic source/filler order required. The pinned
  _encode_fast helper is engineering-only to inspect the actual motherword.
- Real byte-stream roundtrip: bytes(range256)+b'P6LDPC' (262 bytes), actual
  LDPC encode/QAM/identity/APP/BP decode, equality and exact block/pad counts.
  Identity LLR uses nominalN0=1e-6 for finite likelihoods, no channel draw.
- Repeat each actual AWGN10 and Rayleigh10 packet from fresh identical
  RNGs; exact draws/outputs/state repeat required. Report bit/block errors
  and CRC/stream decode failures honestly, without a required favorable BER.
  All attempted uses and actual energy retained even on failure.
- A deterministic prebuilt native-training000000 JP2-rate30 P6SB stream
  may be transferred to Artemis and checked once under identity for both
  digital configurations, complete bytes/decode/native shape. No GT/calib.
  Noise10 checks use synthetic262-byte packet only to bound engineering cost.

Version/module imports and full source identities must be recorded before and
after each check. Evidence paths refuse existing results. CPU tests may run
locally only with this independent compatible runtime; no need to install it
on Mac or sheng. Passing this lock establishes actual digital plumbing only,
not main result/noise performance or a fair learned-vs-digital comparison.

Primary sources:
https://github.com/NVlabs/sionna/tree/v2.1.0
https://nvlabs.github.io/sionna/v2.1.0/phy/api/fec/ldpc/index.html
https://nvlabs.github.io/sionna/v2.1.0/phy/api/mapping/index.html
