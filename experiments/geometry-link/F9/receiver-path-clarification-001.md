# F9 inference executes the native 3D receiver explicitly

Prelock before any native F9 evaluation integration or formal AP. Static review
found that LIGA.forward iterates its full module_list, including MMDet2DHead and
DepthLossHead. F9's training/core interface and intended inference guards forbid
these unused auxiliary branches. Do not weaken that guard after an integration
failure. No such native evaluation failure or measurement has occurred yet.

The evaluation wrapper must execute backbone_3d, map_to_bev_module, backbone_2d,
dense_head and the unchanged model.post_processing explicitly, all in eval/no_grad
with sensor-only input. This is the same native3D receiver chain as F9 training,
with GT absent and postprocessing enabled. Preserve all539 states, including
the484 original detector states; auxiliary parameters remain loaded/read-only.
All four arms use this identical execution path. Do not delete modules, edit
historical source, alter detection/NMS thresholds or make this treatment-specific.
Only received stereo/appearance features enter the receiver. Original-paper
2D/right-image comparator work remains a separate required project condition.

MMDet2DHead writes head_outs and boxes_2d_pred; native3D DetHead/post_processing
use the BEV batch features and3D box/class predictions. DepthLossHead returns
unchanged without depth_gt_img in eval mode. This is a source-level expectation,
not yet a native equality result. During the prelocked48-frame integration,
on the first frame of each of eight conditions, retain the same actual received
backbone/BEV outputs and run a separate receiver-only reference: add the unchanged
auxiliary2D head, then native3D head and postprocessing. Compare every3D box,
score and class to the explicit3D path exactly. This reference makes no new
sender/channel call, uses no fresh noise, no labels/depthGT, no optimization
and no AP. Inspect allowed auxiliary-written keys and verify receiver feature
inputs are unchanged; it is a declared equality diagnostic, not an alternative
formal prediction/fallback. Permit auxiliary execution only inside this separate
reference scope and record its eight calls explicitly. Any mismatch is a blocker
for unchanged-output claims, retained and diagnosed before proceeding.

Formal endpoints execute only the explicit3D chain; the reference is engineering
only. Bind this execution choice/source identity in every formal boundary report
and disclose it in the final method/comparison description. No claim of measured
deployment speed is authorized by removing auxiliary computations.
