# Representation boundary audit after the negative F9 contrast

This is an exploratory design audit, not an experiment protocol or a positive
result. Main baselines and formal SRCNN still have priority.

The frozen native LIGA backbone constructs concatenated stereo features at
each metric-depth node, processes them with3D convolutions/hourglass modules,
and predicts a separate depth probability. The released and current task
configuration selects `use_stereo_out_type: feature`:3D voxel retrieval samples
the multi-channel cost feature, with left semantic features concatenated.
Changing it to `prob` would replace the learned detector input interface.
A probability-only bottleneck cannot be assumed sufficient or compatible with
the existing author checkpoint. Its receiver adaptation and additional compute
would need matched controls and a new native engineering gate.

Sources inspected: local `liga_backbone.py`, `cost_volume.py`, the released
`liga.3d-and-bev.yaml`, and current `stereo_task_seed17_epoch1.yaml`. The actual
metric support comes from `prepare_depth` and the configured spatial range;
coarse raw-cost support has72nodes, prediction support288. Never treat either
as a uniform integer-disparity grid or substitute one for the other. Calibrated
disparity samples are `fu_mul_baseline/depth`, with the corresponding feature
resolution factor. Original/native files were inspected without editing.

The existing GeometryLink already computes a depth posterior from cost features
and supervises it with sparse measured depth. Its communicated content remains
learned multi-channel features. A new posterior loss alone would therefore
change supervision, not establish that useful geometry has been transmitted.
The negative F9 gain-head contrast likewise does not test a probability-only
representation. Both distinctions constrain any following claim.

[Garg et al., NeurIPS2020](https://arxiv.org/abs/2007.03085) already establishes
Wasserstein distribution supervision for stereo and downstream3D detection.
Our proposed metric-depth CDF diagnostic is prior-art-based measurement, not
a new rate-distortion theorem or novelty claim. The source PDF was read in
Sections2–3; other newly located2023 cost-volume papers had inaccessible PDF
fetches and are not treated as fully read or implemented comparisons.

Before a new experiment, specify whether the sender communicates feature
content, explicit depth probability, quantiles, or a hybrid. Specify what the
receiver can reconstruct from those bytes/symbols and public calibration.
Charge all geometry computation and signaling, compare equal-capacity generic
features under the same teacher/training exposure, and measure both geometric
distortion and complete3D task performance. Train-only sufficiency/calibration
diagnostics must precede a main claim. No next architecture or AP primary has
been selected by this note.
