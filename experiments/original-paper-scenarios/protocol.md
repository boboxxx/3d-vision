# Original-paper scenarios: contract and engineering lock 001

Locked before new implementation or measurements. This is a scenario contract
and CPU engineering experiment, not an authorization to start a partially
specified main-validation comparison. The main result grid below is fixed;
missing model/LDPC/calibration identities require a separate committed
execution addendum before their first evaluation. Never edit existing frozen
original/F7/final-receiver sources or restart their live jobs.

## Main result contract

Use native KITTI stereo pairs and the public upstream train3712/val3769 splits,
SHA256 b6417a1d9b18c8fdb085128e633d28ff321b7674a6d1b3841b8f43d865b281cb
and 657ac4bcc1e156e5b106a4ca18e1f88e012787ea1d2b5d0adeea97fee903fa86.
Calibration and all model/quantizer/hyperparameter selection are training-only.
Previously inspected clean val results are explicitly disclosed. All3769
predictions must be sealed before GT/evaluation reading; erasures produce empty
predictions and remain in the complete denominator. No clean fallback.

Source-only: JPEG, JPEG2000, SRCNN, ECSIC and the original-inspired RGB model at
nominal compression10/30/50, plus one clean reference. Original-inspired
global/key spatial reductions36/1,36/4,64/16 are architecture specifications,
not proof of physical-rate equivalence. NN source representations use8-bit
except ECSIC's actual entropy-coded stream. Record raw RGB8 source bits,
representation elements, actual source bytes, and both view dimensions.

Channel-only: the30× semantic source, learned channel codec, LDPC2/3+64QAM,
LDPC1/2+256QAM; AWGN at every integer6–18 and Rayleigh at6,8,…,18dB. Digital
semantic input6-bit; range/scales must be calibrated on training and frozen.
Joint: learned source/channel and JPEG/JPEG2000/ECSIC each paired with both
digital configurations,30×,AWGN at every integer6–18. Reuse identical learned
AWGN outputs across channel-only/joint tables; do not rerun/select a noise draw.

Primary detector is author's full Stereo-RCNN; Car E/M/H, left/right2D,
BEV/3D,IoU.5, report R11 and R40 without claiming the author's sampling is
known. IoU.7/R40 and native LIGA are supplementary transfer/modern metrics.
Direct latent method has an explicit compatible downstream detector, and
therefore requires its own matched same-detector RGB baseline. Comparing
different detectors alone does not prove a communication advantage.

RGB quality: global full image and nonoverlapping ROI union, MSE in[0,1],
PSNR=10log10(1/MSE), SSIM implementation/window/boundary pinned in execution
addendum. Empty ROI is undefined, not zero/perfect. Background is an optional
declared extension. Received model RGB stays unclipped for native inference;
quality and uint8 digital codec conversion policies must be explicit.

Channel model: complex Es=1; noise CN(0,10^(-SNR/10)); Rayleigh h~CN(0,1),
perfect receiver CSI, exact ZF y/h. For our disclosed ambiguity-resolution
variant choose independent per-symbol fading (coherence1). Do not truncate deep
fades, substitute MMSE, add an uncharged pilot, or normalize after channel.
Any practical noisy-pilot/block-fading experiment is a separate extension.

ROI metadata travels an ideal reliable side link as in the reference's
error-free assumption. Do not call it physically free: log exact serialized
bytes and separately log supplied protected-link uses/energy. Only total
physical-resource-matched conditions support a claimed communication gain.
Original exact signaling format is unknown. Every payload, framing/CRC,
LDPC source padding, coded length, modulation tail, reliability-link/pilot use,
attempt energy, and failed attempt is charged. Code rate alone cannot replace
an implemented encoder/decoder. Nominal4 bits/use excludes those overheads.

New-model comparisons require matched training examples/updates/supervision,
same-detector endpoints and actual rate/energy. Current F5b/F6b/F7 and83-epoch
variant are internal prerequisites, not this completed main experiment. New
geometry/risk/allocation controls and seed list remain subject to a subsequent
mechanism protocol; this lock does not manufacture their result or novelty.

## Engineering experiment now allowed

Implement isolated NumPy/Pillow contracts outside all frozen trees:
1. Enumerate the above conditions with unique reusable channel-only/joint keys.
2. Complex AWGN and ideal-CSI Rayleigh/ZF, explicit dedicated PCG64 noise/fading
   generators, unchanged global RNG, exact replay/attempt evidence. Provide
   normalization separately; identity consumes no random draws.
3. Digital block accounting takes explicit LDPC(n,k), QAM M, actual source bits,
   and separately supplied reliable overhead resources. Return actual padded
   block counts and symbol tail; reject wrong rates/nonfinite energy. This
   is accounting only, never a simulated LDPC result.
4. Encode native uint8 RGB pairs with Pillow JPEG or boxed JP2. Fixed caller
   parameter, JPEG quality1–95,4:2:0,optimize=false,progressive=false; JP2
   single rates layer>=1,irreversible=true,MCT=1. Serialize both codestreams
   with explicit20-byte header (magic/version/codec/reserved/lengths/CRC32).
   Record every byte; verify full shape/decode and reject corruption. No
   per-frame AP/GT/PSNR-based quality search, no resized source.
5. Regression tests cover full-noise replay/global RNG, theoretical complex
   noise/fade power and ZF algebra including near-zero channel, incorrect
   rate/padding/overhead, real JPEG/JP2 roundtrip/framing/tamper/native geometry,
   and exact scenario grids. Independent assertions must not merely mirror
   implementation statements.

Run engineering locally and on sheng CPU. Use exactly fixed training frame
000000 as an optional native image smoke test, no labels/calibration/model/GPU
or main validation reading. JPEG quality50 and JP2 rates10/30/50 are fixed
before results; report measured bytes/ratios honestly, not exact target claims.
Pillow builds/version differences are recorded, not assumed bitwise identical.
Store unique evidence IDs, source/protocol/file SHA, command/log/actual native
arrays/resources; refuse overwritten outputs. No new training/AP is launched
by this engineering step. Main execution must pass complete missing-identity
gates through its own future locked addendum.

## Comparator recovery

ECSIC official repo: https://github.com/mwoedlinger/ecsic; paper:
https://arxiv.org/abs/2307.10284. Retrieve/pin source and inspect actual entropy
encoding and published weights before choosing a KITTI implementation.
Pillow behavior source:
https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html.
SRCNN and LDPC source/weights/standards remain unresolved; no substitution is
silently called the original baseline.
