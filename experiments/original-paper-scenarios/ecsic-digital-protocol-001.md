# ECSIC actual source bytes through the audited digital link

Prelock before new framing implementation or packet observations. This engineering
step tests whether the complete sealed ECSIC entropy container can traverse the
existing actual NR-LDPC/QAM chain without a transmitter-length side channel.
It does not measure detector AP, select a rate/checkpoint, train on KITTI, or
establish performance at the original paper's nominal30x source compression.
Preserve every historical scenario/digital/framing/entropy source and active F8
source tree. New implementation lives only in ecsic-digital-code.

## Fixed source bytes and dependencies

Use exactly the two stageB-002 source.p6ec files, once per listed condition:

| Source | Bytes | SHA256 |
|---|---:|---|
| synthetic32x64 |813|acbc2be75dc60f74903dc19c57d100f2ea9772a9c4606f3fb7681e75a632a35f|
| training000000 |44824|e7242cca0a0edd43dbda26b8892894967a75690efeb1716ee538371c1905eb4a|

Require the successful terminal stageB audit, SHA256
7fb7ca99b75054e7e099e2435a05ff94dff00fe1e8896ba320ed83d86bd99524,
and stageB manifest e098d9ec03e7783746ee9a150d0caf2b71e2c82816a47b088b3c4e6e54e6b867.
The fixed source contains all four entropy streams and the166-byte P6EC
header/CRC. Do not substitute its NLL/finite-CDF estimate, strip its metadata,
reencode it, or supply its residuals/scales to the receiver. Public model/config/
CDF identities and dimension rules remain those of the sealed entropy codec.

Reuse DigitalTransceiver unchanged: k1296,n1944,64QAM or k972,n1944,256QAM;
RV0, APP demapping,20-iteration double-precision boxplus-phi BP, Gray QAM,
population Es1, actual per-packet sum|x|^2. Use the existing independent Artemis
CPU digital-cpu-001 environment (torch2.9.1+cpu, Sionna2.1.0), matching installed
PHY sources and frozen dependency inventory before/after. Do not run ECSIC on
Artemis or install/change dependencies in any current environment.

## Outer framing and receive-only contract

Add a separate P6SB v2 codec3 parser/serializer, big-endian20-byte `>4sBBHIII`:
magic P6SB, version2, codec3, reserved0, complete P6EC byte length, second length0,
CRC32 of the complete P6EC payload. The second field does not describe a right
image: this is a single jointly coded stereo container. Leave v1 JPEG/JP2
unchanged; v2 rejects other versions/codecs and nonzero second lengths.

All20 outer bytes traverse the SAME LDPC/QAM chain and are charged. Receiver
gets only actual decoded binary information blocks and publick in{1296,972}.
Validate block width, finite binary values, nonempty capacity and maximum
capacity for a32MiB P6EC plus20-byte header. Derive length ONLY from decoded
header. Require bounded payload length, enough capacity, exact ceil(8*wire/k)
observed block count, all-zero residual padding including nonbyte tails, outer
CRC, then complete sealed P6EC parser validation before returning bytes. Reject
unknown metadata identities, malformed dimensions/descriptors, wrong inner CRC
or excess bytes. No original source length, source hash, source bytes, NPZ
latents or RGB is an allowed receive argument. CRC is error detection, not a
guarantee against undetected errors. Radio frame boundaries and public code
configuration remain ideal assumptions with no synchronization waveform claim.

Discard DigitalTransceiver's legacy length-trimmed recovered return entirely;
its source-length/error record is offline diagnostics only. Framing/container
failures erase the whole pair with no returned bytes or image fallback. Missing
dependencies, altered source/CDF/runtime, numerical/physical implementation errors
and filesystem errors terminate the job, rather than becoming favorable packet
erasures. After successful outer/inner parsing, save the received P6EC bytes for
a later separate fresh process on the original sheng CPU neural runtime. No
neural reconstruction/cropping/detector claims are made by this engineering job.

## Locked packets, RNG and evidence

Order source synthetic32x64 then training000000; within each source order
ldpc_2_3_qam64 then ldpc_1_2_qam256; within each code order identity10,
AWGN6,AWGN18,Rayleigh6,Rayleigh18. Exactly20 one-attempt packets. For packet
index j=0..19, initialize separate dedicated PCG64 noise seed1911+2j and fading
seed1912+2j. No global NumPy/Torch RNG changes; identity consumes neither stream.
Use nominal complex N0=10^(-SNR/10), iid Rayleigh CN(0,1), perfect CSI, exact
ZF with variance N0/|h|^2, no fade clipping/pilots/retry/ARQ/best draw. Identity
uses existing finite demapper variance1e-6 only. No required successful count
for noisy packets; failures and their attempted energy/uses remain evidence.

Save source and outer wire SHA, source bytes, all coded/source-padded/decoded/
LLR/transmitted/received/noise/fading/effective-variance arrays, RNG before/after,
outer+inner reception evidence, actual uses/energy and erasure reason. Save each
successful received P6EC separately; save no received payload on erasure. Require
exact wire/payload equality only for identity; source comparisons on noisy
packets are diagnostic and may not determine acceptance. Run paths refuse
overwrite; preserve partial/failed manifests and do not restart automatically.

First local pure NumPy checks use the two sealed actual P6EC files and bothk.
Independently parse valid fixtures, verify counts/hashes, and reject malformed
outer magic/version/codec/reserved/zero or overflow length/second length, wrong
block width/count, nonbinary/nonfinite information, nonzero byte/nonbyte padding,
outer CRC, and correct-outer-CRC malformed inner CRC/identity/dimensions/order/
lengths/append/truncation. CDF/source dependency failure must raise fatally rather
than return an erasure. Preserve all fixture arrays and check results. Static
compilation/import-independent inspection of the Artemis runner precedes remote
review. A later separately reviewed launch uses CPU4/memory16GiB/time1h; no GPU.
Fresh actual-terminal Slurm evidence and independent full packet audits are
required after execution. Local framing checks alone do not close this stage.
