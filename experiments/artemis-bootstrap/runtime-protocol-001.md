# Isolated Blackwell runtime setup 001

After hardware-only job11423895 completed, observed RTX PRO6000 Blackwell
Server Edition97887MiB,driver610.57.04. Default Python3.12.8 has no torch;
the conda registry's cuda128 prefix does not contain an executable and is not
usable. Do not repair or install into existing/unrelated environments.

Create /mnt/nfs2/engdes/wc296/paper6/envs/torch-cu128-001 with python3 -m venv,
no system-site-packages. CPU-only short partition job,4CPU/16GiB/1h. Install
torch2.7.1 from official cu128 index; numpy1.26.4,Pillow10.2.0,PyYAML6.0.2
from PyPI. Pin this deliberately older compatible runtime, not a latest-version
claim. Official Blackwell support documentation:
https://pytorch.org/blog/pytorch-2-7/ and
https://pytorch.org/get-started/previous-versions/.

Record package versions, wheel hashes from pip reports, full freeze, pip check,
CPU module imports and exact runtime-script/protocol identity. Installation
work is scheduled on a compute node, no GPU reserved during downloads. Failures
remain at unique prefix/evidence IDs; never replace another environment.

After setup success, submit a fresh30min1RTX/8CPU/32GiB enginf job using that
absolute Python, unchanged GPU probe.py, new evidence002. Require actual
sm120 support and finite GPU matmul/conv3d backward. This checks the runtime,
not migrated legacy detector CUDA extensions, native KITTI training or AP.
Both later operators and migrated dataset/model identities need further gates.
