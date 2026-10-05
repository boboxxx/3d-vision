# F4 uniform codec reconstruction warmup (lock before implementation/launch)

F3 and its identity diagnostic failed useful detection. Full372-frame passive
feature reductions were audited: identity decoder cost NMSE1.2338745,
cosine0.1080732; appearance NMSE1.0656292, cosine0.0349285. Plain pooling and
interpolation at the same strides gives cost NMSE0.5924537/cosine0.6391054 and
appearance NMSE0.1275486/cosine0.9344262. These are reconstruction references,
not optimal bounds or proof of a unique architecture/optimization failure.

Next exploratory question: does direct feature-reconstruction supervision
align the existing codec before any additional pooling/capacity or allocation
change? Start from the full fixed finalF3 checkpoint SHA
cb50b4d4073fb2e5be31533d949a2141a46389436a5080429753de8b038aef7c.
All549 state tensors must load exactly. Freeze all original receiver/teacher,
all35 student tensors, the posterior, sensitivity predictor and generic scorer.
Train only the16 cost/appearance encoder/decoder parameter tensors. Fresh
AdamW, LR1e-4 constant, WD1e-4, gradient clipping10, reject nonfinite gradients
before every update. Engineering/diagnostic weights are never substituted.

Keep exact raw-cost and appearance boundaries, pooling(8,4,4)/4, width4,
uniform power, transmitter normalization and no clean residual bypass. Use
identity channel during this fixed one complete3340-update epoch so that this
stage addresses feature alignment without channel noise. Do not remove the
codec. Set the parent link/backbone to eval mode while retaining gradients for
the16 codec parameters: its CNNs contain no BatchNorm/dropout. This disables
unneeded posterior/task-sensitivity losses and random training SNR draws;
it does not freeze the chosen codec weights.

Loss is mean squared error on the interpolated recovered raw-cost tensor plus
mean squared error on the recovered appearance tensor. Targets are the clean
frozen student's actual same-frame features, detached, not detector-author
features. No GT/LiDAR/depth/task loss or original teacher forward is needed.
Use the unchanged native train-only3340/372 split, full native paired image
preprocessing/augmentation, batch1/four workers and seed17. Native replacement
sampling rules remain disclosed; train frames may repeat under augmentation.
Only RGB/calibration/image shape/frame metadata enter the backbone after native
sample preparation. Stop its forward pass immediately after the codec using
an explicit loss-carrying hook; never replace the full detector for evaluation.

All533 inactive state tensors, including originalglobal_step, both imitation
normalization buffers and all non-codec link state, must remain bitwise fixed.
Use an independent trainer step counter; do not increment nativeglobal_step
without the native detector task update. All16 Adam states must have3340
actual updates and finite moments. Preserve contiguous per-update loss,
preclip norm, uses/energy, frame identifiers and immutable raw evidence.

Evaluate the fixed final checkpoint on all372 frames under identity and fixed
10dB AWGN, explicitseed17, complete strict native3D detector. Independently
audit predictions/GT/AP, channel energy and new feature-distortion records.
Do not choose a checkpoint or better rerun. No main-validation tuning, extra
epoch, allocation comparison or scientific gain claim within this stage.
Any later joint task adaptation/noise training must use a separately locked
protocol. F1/F2/F3/F4 budgets and pretrained detector exposure remain explicit.

The trainer and independent checkpoint/scalar auditor are implemented. A
separate three-update native GPU engineering run must pass before the formal
launch; its weights are discarded. Both evaluation channels run complete
native prediction/AP and passive feature diagnostics automatically afterwards.
