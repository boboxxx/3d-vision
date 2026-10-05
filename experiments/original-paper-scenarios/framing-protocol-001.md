# Decoded P6SB framing engineering 001

Prelock 2026-10-04 before implementation or new packet observations. Extend
sealed digital engineering83e103c without changing its historical sources or
results. Preserve the pinned Sionna2.1.0 PHY/runtime, LDPC matrices, Gray QAM,
APP/BP settings and actual energy/noise/CSI conventions. This engineering
addendum resolves receiver byte-length signaling only; no detector/AP, learned
quantizer, ECSIC stream, mainval or allocation claim follows.

The source serializes existing P6SB v1 bytes: big-endian20-byte header
`>4sBBHIII`: magicP6SB, version1, codec1JPEG/2JPEG2000, reserved0, positive
left/right uint32 byte lengths and payloadCRC32. This entire header traverses
the SAME actual LDPC/QAM/noisy chain as payload, is never delivered separately,
and contributes to source bits, block padding, channel uses and actual energy.
CRC covers payload only (existing format); header syntax and observed whole
block capacity constrain lengths. No claim of cryptographic detection or zero
undetected-error probability. Physical frame start/end and public code/modulation
configuration remain ideal system assumptions; synchronization/control waveform
costs are not implemented or claimed. No free source byte-count side channel.

Receiver inputs ONLY hard decoded information blocks, their observed dimensions
and public k in{1296,972}. Validate exactly2D binary blocks of widthk, at least160
header bits, MSB-first. Header-derived byte length must fit observed capacity;
observed block count must equal ceil(8*length/k), rejecting extra or truncated
blocks. Remaining decoded information bits must all be zero padding, including
non-byte-aligned tails at k972. Extract bytes solely using header fields and
verify existingCRC32, then existing image codec/schema/stereo-shape decode.
Any framing/CRC/padding/image-decode failure erases the WHOLE stereo frame;
never use transmitter length, original bytes, original image or favorable retry
as receiver input. Retain failure reason and all attempted uses/energy. The
original source is permitted only in diagnostic metrics AFTER reception.

Implement additive framing-code using the existing unchanged DigitalTransceiver
as transmitter/physical/BP provider. Its legacy returned byte-count-trimmed
bytes are discarded; the new receiver is passed only arrays['decoded'] andk.
Any source lengths/error metrics in the old record are transmitter/offline
diagnostics, never receiver arguments. New reception records explicitly list
allowed inputs, header-derived length, padding, erasure reason and decoded wire
SHA; erasures have no image/byte fallback.

First run meaningful local pure NumPy/Pillow checks with fixed small8x12 RGB
stereo pair (left[(y*31+x*17+c*73)%256], right horizontally rolled onepixel):
actual JPEGquality50 and JP2rate4. For each k, validate complete extraction and
codec decode and reject malformedmagic/version/codec/reserved/zerolength,
capacity overflow, CRC damage, nonzero tail padding, missing/extra blocks,
nonbinary input, wrong block width, correct-CRC malformed image and mismatched
left/right image geometry. Keep actual wires/decoded block fixtures and hashes.

Then one Artemis CPU-only Slurm engineering job4CPU/16GiB/1h uses unchanged
independent digital-cpu-001 runtime. Keep all installed PHY and historical
scenario/digital source identities fixed. Use actual native-training000000
90545-byte JP2-rate30 P6SB stream SHAa9e92e1c66e138f35497984e763db1b63b58be54878d5413b08d68571d3912cd
under identity for both code configurations. New receiver must exactly recover
original bytes and same370x1224 RGB shape without receiving source length.
For each code, also send the one fixed small JP2rate4 pair under AWGN6/18 and
Rayleigh6/18. Noise/fading reset dedicated PCG64 seeds1901/1902 per packet;
no repeat/bestdraw/ARQ and no required favorable error/erasure outcome. Retain
all10 actual packet arrays, complete receiver-only input fixture, source wire
and reception result and diagnostic bit/block errors. Preserve global RNG,
unique paths and all source/runtime identities. Actual terminal Slurm/CPU-TRES
and independently replayed framed extraction/channel arrays must be audited
before calling this complete. Existing main framing limitation may be narrowed
to ideal synchronization/publicconfiguration after this closes, never silently
to a fully physical deployed digital system.
