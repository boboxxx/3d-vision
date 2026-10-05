# Final epoch validation and fixed radio comparisons

Additive inference code005 uses unchanged training004 codecs and cut. Admit
only complete44544-update native and local stream audit proofs for that seed,
both tied to identical completed report and actual raw Slurm terminal0:0.
Use checkpoint-{arm}-epoch3.pt, never an early epoch or AP-selected checkpoint.

Run each G/P/S/B and seed17/23/41 on the manifest's entire ordered3769 validation
IDs. No label file, GT, LiDAR, teacher, auxiliary loss or optimizer enters the
inference process. Keep author484 state, final codec state and source/runtime
identity unchanged. Use native crop-only preprocessing and original KITTI
writer with original calibration/image shape. Save every frame's prediction,
wire/noise/fading/baseband and decoded-field identities, no subset AP.

The public common inference noise seed is 2027100000+integer(frame_id), identical
across arms, training seeds and SNR conditions. AWGN/Rayleigh draw noise first;
Rayleigh then draws fading. Sender gets neither noise nor instantaneous CSI.
Channel code is unchanged004; perfect receiver CSI, no fade clipping.
All arms use19 complex symbols/energy19 per stride4 pooled site. Report actual
counts. Identity carries the same charged payload, not a clean bypass.

Primary remains AWGN10 Car3D R40 Moderate IoU0.7 P-B, seedwise and mean, with
all other arm results reported. After primary, fixed final weights cover
identity and AWGN/Rayleigh6,8,10,12,14,16,18dB. Seed variation is descriptive.

Artemis training runtime lacks the author's Numba evaluator. Preserve its
runtime: inference and native text writer on RTX PRO6000; official AP on the
already verified sheng evaluator after full prediction transfer/closure.
Verify both predictions and GT by dataset hashes before evaluation, load GT
only after inference is fixed, report easy/moderate/hard bbox/BEV/3D/AOS at
native strict0.7 R40 plus secondary0.5 R11/R40. No custom evaluator substitution.
Actual inference/AP exit and full local text/record/wire proofs are still
required before calling the endpoint closed. Prepared code is not an executed
or validated AP endpoint; no formal submission claim follows from it.
