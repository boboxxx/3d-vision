# Actual paired waveform verification

Keep all006 complete-array/state/GT/physics/Adam criteria and bounds unchanged.
Add a direct bitwise check that noise and fading match G/P/S/B for every common
step within a seed, rather than relying only on the generator code and schedule.
After all44544 updates require11136 paired steps and33408 cross-arm comparisons.
Dataset/arm orders, source inputs, private-state deletion and training are unchanged.

Prepare a fixed first256 P versus already sealed first256 G physical-record
probe, after actual P records exist. Verify each metadata line and selected
complete actual TX/RX/noise/fading/baseband arrays against the saved array
hashes, actual complex arithmetic and resources. All256 G source metadata
lines must reproduce the prior sealed256-line JSONL hash. No selected node,
frame or channel subset. This probe verifies pairing/physics, not P gradients,
complete P training, full seed terminal, validation/AP or geometric advantage.

Future full training closures should use007 so common-noise claims have actual
array evidence. Preserve all006 successful proofs and original005 failures.
