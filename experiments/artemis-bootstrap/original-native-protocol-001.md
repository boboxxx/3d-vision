# Original model native RTX PRO6000 engineering lock

Runtime002 hardware/matmul/conv3d passed on actual PRO6000 Blackwell96GiB,
torch2.7.1+cu128,sm120. Next execute the existing frozen original21-file
reproduction variant on this host, without changing sheng's live training.

Transfer only original21 sources, their frozen SHA manifest, the official
SpyNet checkpointSHA3d2a1287666aa71752ebaedc06999212886ef476f77d691a1b0006107088e714,
the existing audited3340-frame training ROI records and audit, and their first
fixed frame000000's two original RGB PNGs. No labels/calibration/LiDAR/validation
data are needed for this compatibility engineering run. Preserve exact hashes
from sheng; fixture is not a complete KITTI copy. Keep remote assets within
/mnt/nfs2/engdes/wc296/paper6/assets/original-native-001.

Run unchanged reproduction/cao2025/probe_native.py: two native-shape discarded
updates, stage1epoch1(562 active) and stage5epoch26(718 active), all finite
gradients and inactive states exact unchanged. Existing engineering objective,
AWGN10, ROIs and padding unchanged. This is a GPU/model compatibility check,
not a formal83-epoch run, trained checkpoint, reproduction of AP or cross-GPU
bitwise equivalence. Require exact source21 before/after, original input/ROI/
checkpoint hashes, actual PRO6000 product/sm120/runtime, phase counts, physical
VRAM and native dimensions. Store exclusive outputs, actual SlurmID and logs.

Submit one enginf RTX/8CPU/64GiB RAM/30min job. No idle reservation, shared
environment changes, legacy detector CUDA binary copying or formal training.
This enables subsequent main-data deployment and separatelylocked scientific
experiments; it does not remove those remaining requirements.
