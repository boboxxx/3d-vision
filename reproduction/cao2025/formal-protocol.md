# Full native original-paper reproduction variant: locked before formal training

Use the complete verified semantic/flow architecture, independent per-view
semantic branches, shared-view/independent-global-key TableIX9-channel CNNs,
and the serialized/protected sparse wire protocol. Every30-block stack and the
official strict60-tensor SpyNet release remains present. Record all768 model
state tensors and718 parameter tensors (12,970,798 parameters). These are
explicit variant choices where the original paper is ambiguous; no exact
author code/weights or published detection AP reproduction is asserted.

Use the fixed train3340/holdout372 sensor-only ROI caches, officialYOLOv5n v7.0
car/bus/truck confidence.25/NMS.45. Verify every consumed native RGB PNG against
its recorded hash. Read no GT, depth, LiDAR or calibration during codec
training. RGB is FP32 in[0,1], no crop/augmentation/resize; bottom/right padding
to6 and unpadding preserve pixel origins. Batch1/four spawn workers, each
training epoch visits each scheduled frame exactly once, shuffled with generator
seed17+1000×stage. Global/model/noise RNG seed17; CUDA algorithms are not claimed
bitwise deterministic. No main validation is used for training or selection.

Run all five stages:12 (2 frozen-flow,10 unfrozen),10 key-only,6 fusion-only,
10 semantic fine-tuning (5 hybrid,5 global),45 wireless (25 channel-only
semanticMSE,20 joint RGBCharbonnier),83 epochs total. Exact component learning
rates follow TableXI and stages.py. Adam defaults betas(.9,.999)/eps1e-8,
weight_decay0, clip10, reject nonfinite loss/gradient before update. New optimizer
between stages; retain state across unfreezing inside each stage. Full strict
predecessor checkpoint and completed independent stage audit are required.
Fresh stage1 model uses only the official pretrained SpyNet; engineering weights
never initialize formal training.

Charbonnier uses epsilon1e-3 and elementwise mean over native pixels/channels,
mean over the two views. Masked error includes the zero-error epsilon outside
the mask. Hybrid is masked+.5global. Channel-only MSE averages global-view
terms and nonempty key-view support terms equally. These reductions, stage4
5/5 switch, batch/augmentation and trainingSNR uniform6–18dB are declared
choices not fully specified by the primary paper. Stage5 trainsAWGN only;
10dB final-stage holdout checks use fixed public noise power. All physical
control/data/padding costs and frame erasures are recorded. An erased sample
consumes its epoch slot and skips Adam. Per-parameter counters distinguish
unfreezing, key-decoder absence on empty-ROI channel-only frames and erasures.

Only active channel BatchNorm running statistics may change. SpyNet mean/std
always remain fixed. At each epoch boundary independently compare all inactive
states, audit optimizer/moments/counters, full frame coverage and all raw
scalar/control/energy records. BN encoder counts advance twice per attempted
stage5 frame; decoder counts twice per successfully decoded frame. Before and
after every epoch verify frozen executable/upstream source identities.

Save every epoch, but never select a best checkpoint or extend epochs based on
holdout. At each stage end only, measure complete372-frame fixed loss diagnostics,
preserving training RNG. Formal stage completion is established by the independent
auditor, not self-reported manifest status. RGB quality and native Stereo-RCNN
AP/channel audits remain separate required final experiments. The9-channel
example's mean physical rate is not yet matched to the new method: do not
claim a communication gain from comparing these unmatched settings.
