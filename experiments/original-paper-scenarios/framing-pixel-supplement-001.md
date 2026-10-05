# Post-hoc decoded-pixel reproducibility supplement 001

After actual CPUjob11424046, independently transferred audit6c6b0a1 failed at
native JPEG2000 image pixel SHA, despite exact decoded bytes and channel checks
for that record. Original evidence is preserved. Local Pillow12.2 versus
Artemis Pillow12.3, both OpenJPEG2.5.4; why pixels differ remains unresolved.
Four successful synthetic receptions agree locally, both native identity
receptions differ. Do not silently relax equality or declare all local image
pixels independently reproduced. This supplement is exploratory, not prelocked
confirmation of a small numerical difference.

Before new inspection, capture allSIX successful receptions' actual decoded
information blocks using an independent raw-byte header/CRC/padding parser
and independent Pillow opening in the original Artemis digital CPU environment.
Never import the receiver or use source length/original bytes as receiver input.
Its output must EXACTLY match the originally sealed wire and recorded image
SHA. Save the complete resulting image arrays and decoder/platform versions.
Run one CPU-only Slurm job2CPU/4GiB/15min, unique paths; do not change packages,
original manifests or channel source. Retain source/fixture/code/array SHA and
actual Slurm terminal proof. Failed receptions remain failures.

Then the local independent verifier audits every raw packet/noise/APP/header/
CRC/erasure/resource and the complete supplementary pixel arrays. It verifies
server independent parsing/decoding EXACTLY reproduces original recorded pixels.
Locally re-decode the identical recovered wires and REPORT exact differences
(number/fraction/max/mean per view), with no pass/fail bound selected afterward.
This establishes received-header bitstream integrity and on-platform image
reproducibility; cross-platform native pixel equality is explicitly unverified
if different. Future main comparisons must use one fixed audited receiver
implementation/environment or the same serialized received pixel tensors.
No AP, exact-author implementation or hardware-independent decode claim.
