# ECSIC KITTI source baseline: native GPU and task scope

Prelock before new GPU model or KITTI training observations. Keep all existing
ECSIC CPU coder/reception, original method, SRCNN and cost-field training trees
unchanged. This adds a separately declared KITTI adaptation of the official
ECSIC architecture, not recovered unpublished Cao weights/settings.

First execute exactly two engineering Adam updates, seed17, native training
pairs000000/000003 in that order, lambda0.01. Discard their adapted weights.
Use the official225-state Cityscapes lambda0.01 initialization SHA
e66b55e2f888f5d113ea6c6414640c7486280d6e18838ba77c650720a5636052,
official config and source revision696f4ae4, verified whole KITTI manifest.
Only RGB8 stereo images enter; no GT, labels, calibration, LiDAR, ROI or detector.
FP32 full native images, top-left preservation, replicate right/bottom padding
to multiples32; no crop, resize, augmentation or clean receiver side channel.

Run a separately allocated Artemis RTX PRO6000 (capability12,0), existing
torch2.7.1+cu128 and NumPy1.26.4. Reuse the already verified torchvision overlay
read-only. Copy existing sheng einops0.8.2 pure Python into a separate read-only
import overlay; freeze every source hash. No shared package installation.
The official utils imports wandb for unused experiment-management routines.
Provide an import-only module with init/log/watch/finish/generate_id raising;
do not invoke author training CLI, remote logger or those management functions.
All model/layers/metrics/transforms/utils numerical source remains byte-identical.

Use exact author training forward (noise-quantized entropy, STE decode),
author calc_mse (RGB squared error times255^2) and four calc_bpp values averaged
over views, loss=(bpp+lambda*mse)/(1+lambda). Adam1e-4 betas.9/.999 eps1e-8,
zero decay, norm clip2.5 as official train.py; no scheduler/AMP for this gate.
Save both complete native RGB inputs, both unclipped predictions, all four
latent dictionaries and rate estimates, every raw/clipped parameter gradient,
initial/after-step weights and complete Adam states. Check all finite values,
gradient presence (zero legal), state/parameter counts, actual update identity,
source/dependency immutability, peak memory and actual Slurm exit.

Independent native/local verification must recompute the losses from complete
saved arrays in FP64, verify all actual array hashes, and reproduce every Adam
state transition from actual prior FP32 parameters/moments. FP64 loss bound
64 FP32 epsilon times(1+absolute reference); moment/parameter bound128epsilon
times the arithmetic magnitude, following the already locked cost-field audit.
This is not an independent CUDA gradient oracle or learned-compression result.
Do not call estimated bpp actual bytes or a nominal10/30/50 source point.

After this engineering gate closes, a separate formal execution addendum will
fix KITTI adaptation exposure and lambda operating-point selection using only
training data, complete real rANS containers and unchanged receiver. Final
source-only and LDPC/QAM wireless comparisons require full3769 received-image
caches and both fixed author detectors. Existing Cityscapes results cannot
silently substitute for the KITTI three-rate baseline. Other seeds deferred.
