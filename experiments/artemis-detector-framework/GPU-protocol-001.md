# Complete author LIGA on actual RTX PRO6000: native forward001

Lock before GPU execution. CPU registry imports and native operator fixtures
are closed; full detector GPU execution is not. Use the existing immutable
Python3.12/torch2.7.1+cu128 runtime, installed overlay005 and the same three
independently executed sm120 native binaries. Allocate enginf gpu:RTX:1,
4CPU/64GiB RAM/30min, verify actual RTX PRO6000 capability12.0, release at exit.

Use the complete author484-state checkpoint SHA
3f36af30b686f23d0868482675177b29cc50c2b378577dd12f9092ddb0235b2e,
93,383,301 bytes. Parse using torch.load(weights_only=True) only. Actual safe
CPU parsing confirms prefixes: global_step1, lidar_model110, backbone_3d293,
backbone_2d21, dense_head_2d35, dense_head24. Preserve all484 states, including
the teacher architecture. Use only the existing explicit sparse-kernel layout
adapter and require complete strict key/shape/dtype/finite/value equality;
never drop unmatched tensors or replace the detector with a smaller surrogate.

Load the frozen clean_holdout.yaml model/config and its base data geometry.
Instantiate the native StereoDatasetTemplate with public geometric metadata,
not StereoKittiDataset's label-containing pickle loader; set its public
boxes_gt_in_cam2_view flag from the same config. No dataset items, labels,
LiDAR, targets or GT depth are read. To avoid redundant unrelated checkpoint
initialization, set only feature_backbone_pretrained and teacher
PRETRAINED_MODEL to None during construction. Strict loading of every author
state immediately afterward determines every resulting tensor; record these
initialization-only overrides, original config digest and full loaded equality.
No source edits, architecture/link/student changes or trained adaptations.

Two fixed training frames000000 and000003, original left/right PNGs and public
calibration only, full original dimensions. Same native crop-only320x1280
augmentation/collation from the existing received-image adapter. Entry contains
exactly SENSOR_INPUT_KEYS. Native DDP/NCCL eval/no_grad; forbid teacher forward
and every training/depth/head loss. Check actual two image-backbone and neck
calls, one cost-volume and one3D-head call per frame. Preserve all484 state
hashes before/after each prediction and complete source/runtime hashes.

Save every processed input RGB array, original/cropped P2/P3, native depth
axis, low-resolution depth logits, full predicted depth maps, complete raw
pre-NMS boxes/class scores and all post-NMS boxes/scores/labels for each frame.
These are engineering artifacts, not a mainval/AP evaluation or calibrated
posterior. Capture native input tensor identity with hooks and require finite
outputs at every saved boundary. Record actual GPU/runtime, memory, full
source/checkpoint hashes and Slurm terminal. Check clean local preprocessing
independently against decoded original PNG bytes/calibration, every saved
array and all484 source/loaded/after tensor identities after full transfer.
Teacher inference remains forbidden; this does not verify sparse GPU teacher
training or backward. Later train-only task gradients and original scenario
matrix/multiseeds remain required. Retain failed attempts before minimal repairs.
