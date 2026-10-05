# Declared SRCNN source interface: engineering checks passed

Protocol bb8c32f preceded implementation c5db795 and all interface forwards.
The adapter now supplies actual two-view RGB8 payload bytes, a 32-byte
dimension/length/CRC header, received-header-only native geometry, studio-range
color conversion, unchanged author luminance filtering and unclipped RGB.
This is our explicit adaptation of the missing compression interface;
the source paper's unpublished SRCNN comparator is not exactly recovered.

All four synthetic check families passed on local and sheng CPU, both actual
foreground commands exiting with status0. Three fixed 32x48 pairs give six
received views on each host. Independent header parsing/direct Pillow payload
construction, malformed-byte rejection, primary-color references/inverse,
identity-core composition and altered received-byte causality pass.
Every official model tensor remains fixed: six readonly tensors, no optimizer.
All source and model identities match between hosts and remain unchanged.

The complete transferred-report check passes all six records. Toy source wire
lengths are 932/272/176 bytes for nominal10/30/50, including framing; these
small-image results are not KITTI compression measurements. Both hosts have
the same actual source-wire hashes in all three cases. Local uses Python3.12.8,
NumPy2.4.4, Pillow12.2.0 and Torch2.12.0; sheng uses Python3.10.15, NumPy1.26.3,
Pillow10.2.0 and Torch2.5.0+cu118. Their recovered float output hashes differ;
no cross-host output equivalence or full MATLAB preprocessing claim is made.
The earlier eight-model core numerical comparison remains the independent
mathematical correctness evidence for the author filtering translation.

The fixed model is the author demo's default9-5-5/ImageNet/x3. Selecting it
before these synthetic observations prevents result-based model choice, but
does not establish an appropriate trained decoder for10/30/50compression.
No KITTI, source-quality score, calibration, GT, task model or radio was read.
Training-only adaptation and native complete source-cache/detector execution
still need their own fixed protocol and actual results.

For rate10 on sheng, actual output ranges are approximately[-0.0191,1.0278]
left and[-0.0922,0.9979]right; out-of-range fractions are0.1736%/1.1719%.
They are retained exactly, as prelocked. Neither silently clipped quality
metrics nor a uint8-only cache contract should be substituted for this output.
Future downstream float conversion and metric clipping must be declared.

Evidence: `data/engineering/srcnn-compression-interface-{local,sheng}-CPU-001.json`
and `data/provenance/srcnn-compression-interface-transferred-verification-001.json`.
The latter verifies complete records/source hashes; native byte/color checks
occurred separately on each host, not by local replay of native raw outputs.
