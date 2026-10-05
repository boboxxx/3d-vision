# Actual detection-loss backward on the complete frozen native detector

Next engineering gate, locked before execution. Native forward001 has actual
COMPLETED0:0 plus full native output/input/state closure; full local transfer
continues and remains necessary. This backward check is independent engineering,
not formal training or promotion of a new method. Preserve the same immutable
base runtime, overlay005, complete484 author states and existing sm120 binaries.
No optimizer or parameter update, sparse teacher invocation, source edit or
new package installation. Allocate the same actual RTX PRO6000 singleGPU.

Use the same two fixed training frames000000/000003 and native crop. Freeze
all parameter requires_grad flags and leave the full model in eval; use the
native model directly with a rank0 NCCL context needed by its loss reductions.
DDP is unnecessary for input gradients when all parameters are frozen. Mark
both processed RGB tensors as gradient leaves. Entry still contains exactly
SENSOR_INPUT_KEYS; forbid teacher and model-level training/depth/imitation loss.

After the sensor-only forward has produced native raw3D-head predictions, read
only these frames' training label_2 files for a separate supervised objective.
Map Van→Car and Person_sitting→Pedestrian as original train configuration;
retain Car/Pedestrian/Cyclist, convert camera boxes using the original
boxes3d_kitti_camera_to_lidar(pseudo_lidar=True,pseudo_cam2_view=False), and
apply the native public point-cloud-range corner mask. Append class IDs1/2/3.
No label, target, annotation metadata or LiDAR crosses the forward/encoder entry.
Targets are strictly loss-side supervision, never a sender/receiver input.

Use dense_head.assign_targets after forward and call its original
get_cls_layer_loss plus get_box_reg_layer_loss (including the configured IoU
and direction terms). These are the actual supervised3D detection losses,
not a made-up differentiable output sum. Backpropagate their sum to both RGB
leaves through the full native cost-volume/voxel/head graph. Require finite
losses, at least one positive assigned anchor in each frame, finite nonzero
both-image gradients, all parameter gradients absent and all484 state values
unchanged. Save every input gradient value, transformed target/assignment array,
loss and complete identity/physical runtime metadata. Preserve failed attempts.
No claim about a trained codec, teacher training, calibrated uncertainty,
geometry-specific benefit or complete mainval/AP follows from this gate.
Independent whole-array/target/source/state acceptance and later actual matched
task representation training are still required. Input hashes and execution
code must be committed before submitting any backward job.
