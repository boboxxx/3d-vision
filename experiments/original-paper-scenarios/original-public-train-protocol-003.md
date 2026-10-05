# Original RGB/flow public-training semantic baseline, nominal30, seed17

Prelock before execution. This run closes the original global36/key4 source
architecture on all3712 public training pairs. Reuse the exact frozen
reproduction/cao2025 model, semantic losses, stage schedule, optimizer groups
and official SpyNet initialization. The three other semantic branches remain
complete, including all30-block stacks,768 registered states,718 parameter
tensors,12970798 parameter values. This is a disclosed implementation variant:
authors' communication code/weights, exact split and several reduction choices
are unavailable. Do not label it a reproduced published AP number.

Train stages1/2/3/4 for12/10/6/10 complete epochs,44544/37120/22272/37120
updates,141056 total, batch1 native unaugmented RGB[0,1], source hashes checked
on every read. Worker0 and CPU threads2 are declared execution choices.
Shuffles use NumPy PCG64 seed17+1000*stage+epoch, each complete3712 exactly
once. Seed17 initializes all fresh model/CPU/CUDA RNG. Preserve the original
learning rates, first2 frozen-flow epochs, stage4 fixed5 hybrid/5 global switch,
Charbonnier epsilon.001, masked+.5global hybrid, Adam(.9,.999)/1e-8/decay0,
gradient normclip10. New Adam between stages, same Adam at within-stage
unfreezing. Channel networks exist but remain frozen in these four stages.

Require original-full-train-ROI002 native+local proofs, exact records and whole
KITTI manifest, and the earlier actual PRO6000 full-model compatibility gate.
Read only training stereo PNGs, official SpyNet and locked metadata; no GT,
calibration, LiDAR, detection network, validation PNGs or held-out diagnostics.
The former372 internal holdout is training data here; no holdout loss is
computed and no checkpoint/epochs are selected from outcomes.

First run exactly3 discarded native source pairs in stage1 engineering mode,
using the formal loop, source guard, loss/gradient checks and save format. Close
actual Slurm job and .0 exit0, independently audit all saved state/moments and
all3 records natively, repeat with complete transferred engineering artifacts
locally. Only that fresh engineering closure admits formal stage1. Engineering
weights never initialize formal training. Predecessor stage final checkpoint
and independent native+local audit plus normal actual terminal admit subsequent
stages; don't launch the next stage from an in-progress checkpoint.

Save immutable safe weights_only checkpoints every epoch and the stage's exact
initialization; per-update logs preserve order, hashes, dimensions, loss, active
parameter counts, norm, finite checks and optimizer counts. Independent audit
reconstructs schedule/order, exact source metadata, full frozen states, complete
Adam groups/counters/moments, and initial or full predecessor state. This audit
is saved-state/counter evidence; it does not independently recompute every
forward/backward gradient or Adam numerical transition. Preserve all failures.
Freeze all sources before/after each epoch, no mutation of old runs/runtime.

Final stage4 epoch10 is the fixed source-only nominal30 model. Its8-bit latent
representation calibration/wire/receiver and complete3769 detector endpoints
are separately required before AP. A separately locked stage5 will train
channel codecs25 epochs and joint RGB20 epochs (167040 additional updates),
using the original main contract's reliably transmitted ROI metadata and
perfect-CSI per-symbol Rayleigh at evaluation. The old repetition/CRC/pilot
block-fading transport is a separately disclosed extension, not the main
original channel scenario. Nominal10/50 architecture variants remain required.
This003 run does not complete all83 epochs, three source rates, rate matching,
wireless baselines, the new method, or the manuscript.
