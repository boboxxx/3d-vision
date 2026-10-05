# Fixed single-seed KITTI adaptation of official ECSIC

Lock before any full adaptation, lambda selection, KITTI main reconstruction
or detector observation. The two-step native GPU engineering001 must close
with its actual normal Slurm exit plus native/local complete-array RGB/RD/Adam
proofs. This is a disclosed KITTI adaptation using public Cityscapes pretraining;
Cao's unpublished ECSIC initialization/hyperparameters remain unknown.

Keep official225-state architecture, source revision696f4ae4, model config,
full-image replicate padding, training quantization, source-RD loss and isolated
GPU/import runtime from engineering001. Every candidate starts from the same
unaltered public lambda0.01 weights; never initialize from engineering weights
or another candidate. Do not install shared packages or touch active sources.

Train six lambda candidates:0.001,0.003,0.01,0.03,0.1,0.3. These cover lower-rate
points for physical matching as well as the original nominal10/30/50 source
points. They are operating-point models, all using **one seed17**, not multiple
random-seed trials. No architecture pruning or arbitrary resized demo input.

For each lambda:10 complete epochs of all3712 training stereo pairs, batch1,
37120 actual Adam updates. Per-epoch complete order is PCG64(seed17+1000*epoch)
permutation of manifest training IDs, identical across candidates. Native
RGB8→FP32[0,1], common left/right replicate padding only; no random crop,
flip, augment, detector, depth/label/calibration/LiDAR/ROI or task supervision.
Seed torch/NumPy/CUDA17 once before model construction, preserving identical
quantization RNG exposure/order across candidate models. Author loss is
(mean_view_bpp+lambda*mean_view_RGB_MSE_255scale)/(1+lambda).
Adam1e-4,betas.9/.999,eps1e-8,decay0;clip norm2.5. No scheduler/AMP,
early stopping, validation evaluation, best-epoch or retry-draw selection.

Save every frame ID/source hash/image and padding dimensions, exact update
index/epoch/lambda, RD scalars, RNG identity, all gradient finite/count scope
and norm, all optimizer step counts, plus the complete weight/Adam checkpoint
after every epoch. Formal all-update logs verify exposure/scalars and recorded
native invariants; they do not independently reproduce every CUDA gradient or
Adam transition. Engineering001 supplies the complete tensor arithmetic check.
Do not call formal scalar logs a complete raw-gradient replay.

Only final epoch10 weights qualify for source calibration/reception. After
all candidates finish normally and full records/weights pass audit, encode
the same fixed64 training pairs as the closed JPEG calibration (sorted first64
of geocomm_tune_train) into complete real four-stream rANS containers using
the existing frozen CDF/support/escape/header rules and public final weight
identity. Use fresh decode-only processes; estimated bpp is diagnostic only.
Choose each target10/30/50 candidate by smallest absolute log residual of
pooled raw_RGB8_bytes/full_container_bytes, tie smaller lambda. Report actual
ratio/residual even if target is not reached or targets choose the same model.
Do not duplicate or tune a candidate to hide a missed target. The other three
weights remain available for predeclared resource curves. Code/receiver
identity adaptation and complete calibration must receive their own addendum
before execution; no codec/CDF fitting on validation.

Complete3769 native validation pairs, no clean fallback, are still required
for each selected source point and both fixed author detectors. Radio stage
uses entire containers and actual LDPC/QAM configurations/scenario matrix,
charges all padding/framing/attempts and preserves erasures. Comparisons of
source-training exposure and off-domain public pretraining are disclosed.
Matched same-detector resource/energy endpoints remain required before a
communication-efficiency claim. This protocol supplies a baseline, not a
positive result for the geometry method or statistical significance.

Execution: Artemis RTX PRO6000 array, six candidates with concurrency1,
24h limit/candidate,4CPU/64GiB. Preserve the running cost-field seed17 job.
If a candidate fails/timeouts, retain it and diagnose; never restart blindly
or treat partial training as final. Full candidates222720 updates total.
