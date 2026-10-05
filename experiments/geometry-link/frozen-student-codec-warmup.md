# F3 uniform codec warmup with the audited task-adapted student (lock before launch)

F2's fixed one native task epoch restores uncompressed holdout Car moderate3D
AP_R40 to42.3194602% in the independently recomputed strict372-frame evaluation.
The post-training automatic evaluation differs slightly (42.3157984%); both
records are preserved and repeatability is being checked. No communication
result follows from this uncompressed diagnosis.

Question: can the existing cost/appearance JSCC link preserve this fixed useful
student representation? Freeze all35 student tensors as well as the complete
original receiver/teacher parameters and BN statistics. Initialize all519
original/student states from F2 checkpoint SHA
7e1ceacb4abff0bdb32ef89507b9786e1d80ae9c166ccaaeb4ffb4fcbd639bd2.
Initialize only the link's new tensors freshly at seed17; fresh optimizer
states, no F2 optimizer continuation. Disable the now-constant student-feature
distillation term and its redundant ResNet feature-teacher computation.

Keep raw-cost boundary, nonlinear posterior64→16→1, geometry pooling(8,4,4),
appearance pooling4, complex_width4, uniform unit-mean-energy allocation and
AWGN. Train SNR uniformly[-5,20]dB; evaluate fixed10dB with explicitseed17.
Retain full native3D/depth/2D/LiDAR imitation losses plus posterior sparse-depth
CE weight0.1 and existing native3D-gradient sensitivity distillation weight0.1.
The sensitivity predictor is trained even for uniform allocation to retain
the shared architecture/supervision for later matched controls. Generic scorer
remains frozen/unused. Expected28 trainable link tensors; verify actual scope.

Fixed3340/372 train-only fold, native paired augmentations, native empty-GT
replacement sampling, batch1/four workers, one complete3340-update epoch.
AdamW constantLR0.0001, WD0.0001, clip10, reject nonfinite loss/grad before update.
All initial519 states must remain exactly unchanged except global_step+3340
and the two original training-only imitation normalization buffers. Full saved
checkpoint/optimizer and contiguous CRC scalar evidence must be audited.

Preserve native automatic holdout outputs. Separately evaluate the fixed final
checkpoint on all372 IDs with strict sensor-only inputs, and independently
audit rawGT/predictions/AP plus all channel records: expected62400 complex
uses/frame, mean symbol energy1, CBR0.0260416667; no free sender power/scale/CSI.
No main validation tuning, extra epochs or power comparison within this stage.
If training fails, preserve failure and resume only under a logged repair.

Charge F1's five feature epochs, F2's task epoch, pretrained detector/teacher
exposure and this codec epoch equally to all subsequent matched conditions.
This exploratory warmup establishes only whether the representation survives
the bottleneck. Uniform/geometry/task/geometry-task, generic scoring, matched
RGB, multiple rates/SNRs/pilot-fading/seeds remain separate required studies.
