# Stage B output digest repair, prelock

First byte-only receiver attempt ecsic-entropy-stageB-001 failed on synthetic32x64
after saving decoded arrays. Its source-input audit hook correctly rejects all
NPZ reads, including the program's subsequent attempt to hash its own output
NPZ. No training000000 entropy encoding or native measurement occurred. Keep
failed manifest, source, receiver log, payload and server reconstruction intact.
This is an output-evidence plumbing bug, not a decoded-equality outcome.

Repair only output serialization: write arrays to an in-memory BytesIO NPZ,
hash those bytes, then write the same bytes once. Do not permit any NPZ reads
inside the receiver. Add a subprocess regression with the actual helper and
an NPZ-read rejection hook; external parent verifies written bytes/hash and
arrays. No change to CDF, entropy encoder/decoder, inputs, neural dependencies,
checkpoint, resource accounting, equality requirement or thresholds. Existing
204-case and independent C++ byte gates remain required. Rerun the same two
fixed fixtures in new unique ecsic-entropy-stageB-002 only after this regression.
