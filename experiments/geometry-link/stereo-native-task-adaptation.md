# F6: fixed clean3D task compatibility of the F5b stereo-feature codec

Lock before implementation/sanity/formal execution. The complete F5b run
stereo-feature-native-seed17-001 and source closure d3c788e finished3340
reconstruction updates and both predetermined full372 evaluations. CarModerate
3D AP_R40 is1.1551948753% identity and .1333333333% AWGN10dB, versus the
previous fixed uncompressedF2 reference42.3194602%. All full-state, optimizer,
scalar, AP/GT/prediction/channel/feature/source audits passed. This is not a
useful wireless detector or a fair superiority result. It motivates an explicit
clean task compatibility test before allocation or channel adaptation.

## Sole initialization, architecture and scope

Initialize every535 state from the sole finalF5b checkpoint SHA
9a4b291070df160e6c03e75ed31eeee1dda7f6bfbaa0528a43a9cdf526977553,
located at /mnt/d/paper6/runs/stereo-feature-native-seed17-001/checkpoint_epoch_1.pth.
No engineering weights, new model states or partial load. FreshAdamW discards
allF5b optimizer states. Keep exactly the existing16 codec weight/bias tensors
trainable; all519 F2 predecessor states stay bitwise fixed, including native
global_step and imitation normalization buffers. Preserve the exact F5b
stereo pool16×1,32→2realcode, shared decoder; appearance pool4×2,32→8realcode.
Native input is320×1248, stereo32×320×1248×2views andappearance32×80×312.
Total remains62400complex uses, joint average complex energy1, fixed public
receiver geometry/layout, zero identity-channel noise, no pilots/control.
No private clean scale, power, code, feature or true channel coefficient enters
receiver decoding. This is clean task adaptation, not noisy wireless training.

## Native3D objective with teachers excluded

Parent, student, codec and every original receiver module stayeval; gradient
calculation remains enabled for the16 codec parameters and intermediate
receiver activations. Do not use the legacy semantic_link string filter.

Explicitly run native stereo backbone, height-compression/map-to-BEV andBEV
backbone in original order using RGB/calibration/image/frame inputs plus the
native paired augmentation's random_T when present. This optional training
geometry transformation must be preserved and logged; it is not an inference
side channel. No GT boxes, depth target, LiDAR input, labels or clean feature
may enter the sender or communication computation. After those receiver
features are fixed, attach the same native augmentedGT boxes to the native3D
head. In eval mode with GT boxes present, it assigns original anchors/targets.
Use native3D head get_loss: classification plus box/direction regression with
unchanged configured weights and IoU term. Head cache is cleared natively each
forward. Do not use model.get_training_loss or broadmodel.forward.

Original ResNet feature teacher, original feature neck, LiDAR teacher, depth
loss head,2D detection head and imitation supervision may not execute. No
student feature distillation, reconstruction term, posterior/calibration loss,
sensitivity distillation or receiver adaptation. No model global-step update.
Observe actual channel-before-build_cost chronology, received-only cost and
appearance inputs, student2/channel1/codec1/build_cost1/map-to-BEV1/BEV1/3Dhead1
per update, forbiddenteacher/depthloss/2Dhead0. Verify targets first become
available at the3D head, not sender/backbone. Additional construction-only
weights remain in the complete535 checkpoint and frozen even when unused.

## Fixed budget and engineering gate

Seed17, one fullnative original-train3340 epoch, exactly3340 optimizer updates,
batch1/four spawnworkers. Native paired augmentation, GT-dependent filtering
and empty-GT resampling remain and are charged. Record actual returned IDs and
coverage rather than assuming scheduled uniqueness. FreshAdamW constantLR1e-4,
WD1e-4, defaultbetas(.9,.999)/eps1e-8, gradientclip10. Reject missing active or
nonfinite loss/gradients before each update. Use an independent update counter,
not original global_step. Save every535-state initialization and sole final
epoch1 checkpoint, complete16-parameter optimizer and raw contiguousJSON rows
with native3D loss components, total loss, LR, preclip gradient, GT-at-head
boundary/shape evidence, channel count/energy/CBR and executed module counts.

Commit implementation before separate3-real-update nativeGPU engineering and
independent audit. Verify gradients through the nativeCUDA stereo cost operator,
all16 actualAdamstep3/moments/changed weights, all519 inactive states exact and
all535 layout/finite values. Engineering weights never initialize formal.
Measure full task-backward GPU memory while originalPID26642 continues; use
NVIDIA physical memory plus2GiB margin, not WSL per-process mem_get_info alone.
If unsafe, wait for a suitable GPU window without reducing native resolution,
modifying/killing original training or silently changing the budget.

Commit engineering evidence before formal launch. Freeze project/receiver
executable sources through formaltraining, finalevals andindependentclosure;
original reproduction/cao2025 sources/protocol remainfrozen throughout its
separate83-epoch cycle. Hash allfour source trees before/after. Independent
audit validates exactinitialization/fullstates, all16 actualAdam3340 counters,
finite moments and changed weights,519 frozen values, fixedLR/objective/boundary,
fullraw scalar membership/reductions/counters and actualcalls.

## Predetermined evaluation and subsequent research

Use only finalepoch1 checkpoint for orderedfull372 sensor-only nativeidentity
andAWGN10dB evaluation, seed17. Keep fullnativeAP_R40 IoU.7/prediction/GT/calib,
channel energy/count/control, passive3-feature distortion, full535 load/read-only
state and actualteacher-free receiver boundary audits. Report both, all prior
F1/F2/F3/F4/pooling/F5 exploration, and nativeauthor-pretraining holdout limitation.
No bestcheckpoint, extraepoch, allocation tuning, newnoise seed ormain-validation
selection. Inspect clean task improvement and the remaining noise gap separately;
failure does not establish a fundamental rate bound or a unique cause.

Any later wireless adaptation, receiver adaptation, rate change, uncertainty
calibration or importance study requires a separateprecommitted budget and
matched controls. Charge all F1/F2 student exposure, F5b reconstruction, this
3D GT adaptation, predictor supervision and selection exploration equally.
Do not mutate the running original-paper RGB reconstruction protocol; add a
matchedGT-adaptation control when necessary for fairlatercomparison. Useful
model, calibratedpreventabledamage, physicalresourcefrontier, generic/shuffled
controls, fading/pilots, multiseed mainvalidation, noveltyaudit andmanuscript
remain mandatory project work. F6 completion alone cannot complete the goal.
