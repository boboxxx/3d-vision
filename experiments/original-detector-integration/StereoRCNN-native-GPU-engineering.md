# Native Stereo-RCNN GPU received-RGB endpoint engineering

Prelock: first ordered held-out frame 000036 only, seed 17, three endpoints
clean RGB relay, full original fresh 768-state initializer identity, same
initializer AWGN 10 dB. No optimizer or detector fine-tuning, no AP evaluation,
no threshold/condition/checkpoint selection. This completes engineering of the
second detector endpoint and cannot establish a trained communication baseline.

Use branch-1 author Stereo-RCNN ResNet101, exact full 670 saved states, checkpoint
`/mnt/d/paper6/checkpoints/stereo-rcnn-branch1-author.pth`, SHA256
`b7a07a897224f75bc30b2d4c06f39927e92933a8bcaad6e679aff605b2346c14`.
Strict full shape/dtype/finite/value equality, preserve native evaluation flags,
all modules eval/no_grad and all 670 states unchanged before/after each endpoint.
Use the verified compiled author operator binary with SHA256
`5c3cbc235cad8e67dbb440cb01940d912f72b702455e51dafde2573f7d9c6429`.

Original codec runs on CPU, two threads, fresh stage-1 initializer SHA256
`f4d0d5d5bcc203ee5fd0554a0e691b2a530c56cb55d48fe21b6fc0f97331d3cb`,
all 768 states read-only, cached sender-only YOLO boxes, actual wire/control and
observation-only receiver, Generator17 per condition. Never use broad forward
or clean-image reconstruction loss. On erasure preserve all physical uses and
emit empty prediction/no detector call. No GT/calibration enters codec.

Received float RGB goes through unchanged official BGR/mean/crop/scale blob
preprocessing, without clamp/uint8. Both detector backbone and dense photometric
alignment must receive the actual resulting received-image tensors. Record
input identity hooks; use existing complete official 3D postprocess at fixed
confidence .05, with public calibration and raw shape. Native zero-valued GT
arguments are unused placeholders in this released evaluation API, never labels.
Keep threshold and solver unchanged. Record finite prediction rows and native
2D/initial/dense solution counts; a zero-detection result is valid engineering.

Schedule after F6b terminal closure. Require single NVIDIA GPU physical free
>=12 GiB initially and measured peak reserved+2 GiB <= initial physical free.
Original PID26642 and all other users' jobs remain untouched. All root executable,
four project/upstream source identities and original21/protocol remain unchanged
through this probe; implementation is isolated in this directory. Native author
model package must not collide with original model.py; load Cao model by alias.
Preserve failure manifest/log and use unique output ID for any repair. Never use
these untrained/engineering weights as a formal initializer. Formal detector AP
requires the final fully audited stage-5 epoch45 codec and fixed full 372 frames.
