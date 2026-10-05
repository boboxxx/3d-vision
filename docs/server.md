# sheng execution record (2026-10-03)

SSH works at `sheng@100.94.183.27` after Tailscale authentication. Host: WSL2
Ubuntu 20.04.6, RTX 4090 (24564 MiB), i9-13900K (32 threads), 31 GiB RAM.
Observed kernel: 6.18.40.1-microsoft-standard-WSL2. WSL root has about 24 GiB
free; D: about 5.1 TiB free. Code: `/home/sheng/paper6`; large assets:
`/mnt/d/paper6/{data,checkpoints,runs}`.

## Runtime actually built

Created `/home/sheng/paper6/.venv` with `--system-site-packages` from existing
`/home/sheng/anaconda3/envs/epiu-dsgn`. Base PyTorch is inherited read-only;
new dependencies are installed in the project venv. Other projects/shared
packages were not changed. This is not a self-contained container. Package
versions and exact source-file hashes are saved with engineering evidence.

- Python 3.10.15; torch 2.5.0+cu118; torchvision 0.20.0+cu118.
- Existing CUDA compiler 11.8.89; g++ 9.4.0; architecture 8.9.
- Compiled MMCV-full 1.7.2 and all three actual LIGA CUDA extensions.
- spconv-cu118 2.3.8; NumPy 1.26.3; Numba 0.59.1/llvmlite 0.42.0 in the venv.
- cuda-python/cuda-bindings 11.8.7; `NUMBA_CUDA_USE_NVIDIA_BINDING=1`.
- Custom mmdetection revision `5cf3d2227531101dc45ea9f5b4f8c04ee124afcf`.
- LIGA revision `aee3731a24a0ab1667e633e520cc89be2f135272` plus saved patches.

Modernization preserves the full network and sparse kernel values. It replaces
removed THC/NumPy/PyTorch APIs, sparse feature mutation and kernel layouts,
and allocates NMS labels on the correct device. Inference skips the training-only
LiDAR teacher and computes depth metrics only when GT is supplied. Training
retains the teacher. The upstream test seed honors the launcher's rank seed.

```bash
cd /home/sheng/paper6
source scripts/sheng_env.sh
python scripts/patch_liga.py --root third_party/LIGA-Stereo
python scripts/modernize_liga.py --root third_party/LIGA-Stereo \
  --mmdet third_party/mmdetection_kitti
```

The environment script selects project Python, existing CUDA, one GPU and
the concrete WSL driver loaded by working PyTorch for Numba. The default
ctypes binding crashes with PyTorch's active CUDA context in this WSL session;
the NVIDIA binding passes the original evaluator fixture. It bounds CPU threads
and uses `MAX_JOBS=2` and
`TORCH_CUDA_ARCH_LIST=8.9` for compilation.

## Engineering evidence

Independent numerical checks passed for actual cost-volume interpolation and
gradients (float32/float64, strides 1/2/4), legacy sparse kernels versus dense
convolution, voxel coordinates/storage, analytic IoU/NMS and point membership.
A synthetic perfect-detection fixture passed for the original KITTI evaluator.
It is engineering evidence, never a KITTI benchmark result.

Downloaded the author's checkpoint from its official README link to
`/mnt/d/paper6/checkpoints/liga-author-download`: 93,383,301 bytes, torch ZIP
format (do not unpack). SHA-256:
`3f36af30b686f23d0868482675177b29cc50c2b378577dd12f9092ddb0235b2e`.
Strict audit matches all 484 state tensors of detector and LiDAR teacher.
Expected ImageNet partial-initialization warnings precede a complete LIGA load.
The bundled separate teacher checkpoint is retained in upstream `ckpt/`.

Complete 320x1280 synthetic forward passed for clean detector and random new
codec. Complete training loss/backward passed for all 18 active codec tensors
with detector frozen and no optimizer step. Inference omits LiDAR/GT and asserts
zero teacher calls. Width-4 AWGN accounting: 64000 complex uses, energy 64000,
CBR 0.0260416667 on this shape. This establishes execution, not accuracy/gain.
Raw attempts, including failures, are in local `data/engineering` and remote
`data/`; logs are `/home/sheng/paper6/logs`.

```bash
python scripts/verify_liga_runtime.py --output data/runtime-new-attempt.json
python scripts/verify_full_model.py \
  --checkpoint /mnt/d/paper6/checkpoints/liga-author-download \
  --calibration /mnt/d/paper6/data/kitti/training/calib/000000.txt \
  --forward --output data/full-model-new-attempt.json
```

Use unique output names. For codec backward add
`--config configs/liga_geometry_awgn.yaml --codec-gradient`.
The verification explicitly permits untrained link tensors for engineering.
Actual benchmark evaluation rejects incomplete detector or link checkpoints.

## All five KITTI inputs verified

The existing downloader obtains five public KITTI S3 archives into
`/mnt/d/paper6/data/kitti/archives`. It uses resumable ETag/size checks,
single-part MD5 where applicable, SHA-256 and extracted-member CRC checks,
and requires 7481 files in each training folder. All labels/calibration are
verified, as are all 7481 files for each stereo view and Velodyne. The downloader
finished and exited successfully. Velodyne SHA-256:
`2a005138e7a7c01eda7abe0ba48139b27d864984f1c1cab38f3df1c54e87eb28`.
See `logs/kitti-download-ranges.log` and dataset
`download-manifest.json`. Do not start a duplicate while this job is active.

After measuring faster four-connection S3 transfers, the downloader was migrated
from PID 2085 to PID 7522, preserving both contiguous partial archives. Saved
`data/engineering/download-range-migration.json`. It now uses four verified
disjoint range connections per archive and a process lock. This was a measured
throughput change, not a timeout-triggered restart. Eleven local tests pass,
including real HTTP range/resume/corruption checks.

Locked standard split from official OpenPCDet revision
`233f849829b6ac19afb8af8837a0246890908755`: train 3712, val 3769, disjoint and
covering all samples. SHA-256 train:
`b6417a1d9b18c8fdb085128e633d28ff321b7674a6d1b3841b8f43d865b281cb`;
val: `657ac4bcc1e156e5b106a4ca18e1f88e012787ea1d2b5d0adeea97fee903fa86`.
This does not establish the original semantic paper's exact split.

After download state is `finished`, generate infos with full paired-image and
calibration audit first. The wrapper uses upstream train/val operations without
unused testing-data or GT-database generation.

```bash
python scripts/prepare_liga_data.py --root /mnt/d/paper6/data/kitti \
  --output data/kitti-preparation-001.json --workers 4
```

LIGA `data/kitti` points to that dataset; `outputs` points to
`/mnt/d/paper6/runs/liga`, protecting the nearly full WSL root.

## First substantive run after data/evaluator validation

Evaluate the complete clean checkpoint on every validation frame before codec
comparisons. The full original downstream Stereo-RCNN clean run completed;
LIGA clean evaluation is running with verified train/val infos. No training has begun.

```bash
python -m torch.distributed.launch --nproc_per_node=1 scripts/run_liga.py test \
  --seed 17 --manifest data/runs/clean-seed17.json \
  --cfg_file third_party/LIGA-Stereo/configs/stereo/kitti_models/liga.3d-and-bev.yaml \
  --ckpt /mnt/d/paper6/checkpoints/liga-author-download \
  --save_to_file --launcher pytorch --eval_tag clean_seed17
```

Manifests hash actual source files because rsync excludes Git metadata. Retain
recorded upstream patches/revisions. Test outputs go beside the checkpoint in
its `.eval` directory. Confirm metrics, all 3769 predictions and finite outputs
before calling this a verified clean baseline. Original Stereo-RCNN/RGB codec
reproduction remains a separate required comparison.

Only after clean validation, fix matched warm-up/joint budgets and train codecs.
Use `--init-ckpt` for initialization; `--ckpt` resumes strict model/optimizer
state. Do not resume a legacy optimizer with unconverted sparse moment tensors.
Test trained communication weights. Upstream 60 epochs is not a selected codec
schedule. Both clean detector AP references are independently audited;
scientific communication hypotheses remain unverified.

Matched RGB and raw-cost geometry configurations pass full 320x1280 forward and
task-loss backward on the GPU, with all detector weights retained. See
[compute audit](compute-audit.md) for measured sender costs. The raw-cost split
only moves original dres0/dres1 to the receiver; it does not remove these layers.

Branch/checkpoint audit: the first Stereo-RCNN full run is compatibility
evidence, since master epoch21 and branch1 epoch13 weights differ. Formal
clean validation uses branch1 code and its own released checkpoint; see
[baseline record](stereo-baseline.md). No original semantic reproduction is
implied by either run.

## Student encoder and coupled posterior

Optional student geometry config passes complete 320x1280 forward and full-task
backward with all 484 original detector tensors loaded, finite gradients for all
55 trainable student/codec tensors, and no optimizer step. Original feature and
LiDAR teachers execute zero times during inference. Corrected raw posterior
head uses nonlinear left/right coupling. All four local/deployed source tree
digests match in `*-source-manifest-student-coupled.json`; original patches and
earlier linear variants remain preserved. Thirteen local tests pass.

Full KITTI audit and actual upstream train/val infos generation started as
`kitti-preparation-001`; log `logs/kitti-preparation-001.log`. No codec training
or original communication reproduction is implied.

Optional task sensitivity config `liga_student_geometry_task_awgn.yaml` passes
full GPU inference and complete native training loss backward, 63 finite new
parameter gradients, zero inference teachers and 484 matched original tensors.
All four deployed trees match `*-source-manifest-student-task.json`. Fifteen
local tests pass; inference uses predicted importance and training-only native
3D loss gradients are detached. No codec is trained yet.

## Completed data preparation and real clean evaluation

`data/kitti-preparation-001.json` records the completed all-frame stereo pairing,
calibration checks and actual upstream infos generation. Train3712/val3769,
order matches locked split. Train info SHA e330b8d865ea4c7d89e15e9ef8d75f06d0dfabd668ac89081847dd720050da34;
val SHA317cd48d85ff8accd43aa56fd9556472b23a2e93c7cba8166694acb64a719f81.
All five archives have CRC/size checks for7481 training files per folder.

Real full LIGA clean run003 finished all3769 frames, seed17,
`data/runs/liga-clean-seed17-003.json`, log `logs/liga-clean-seed17-003.log`,
checkpoint `liga-author-download`. All484 tensors load strictly. Batch1,
4 spawned data workers, all3769 validation frames. The model boundary accepts
only batch size, stereo RGB, camera calibration, image shape and frame IDs.
GT recall is computed after predictions; AP uses the unchanged evaluator.
No LiDAR or labels reach detector forward. Metrics and predictions are retained
in `/mnt/d/paper6/checkpoints/liga-author-download.eval/eval/epoch_6/val/clean_seed17_003`.
The epoch_6 directory label comes from legacy filename parsing; actual author
checkpoint metadata remains epoch53. Do not treat the directory as epoch selection.

Two earlier zero-prediction attempts are retained: old explicit port18888
mismatched elastic agent29500, then wrapper relative `tools/test.py` failed
spawn resolution against ORIGINAL_DIR. Controlled interrupts have separate
JSON records. Fixed launcher-provided env rendezvous and absolute runpy entry.
Four source trees match frozen `*-source-manifest-sensor-spawn.json`.
Seventeen local tests pass. Failed manifests/logs and historical script overrides
are saved; no failed initialization is a benchmark result.

The independent auditor passed; the invocation below records its working
directory (do not overwrite its existing output). It ran from the same
upstream working directory (this run records dataset root as `data/kitti`):

```bash
cd /home/sheng/paper6/third_party/LIGA-Stereo
source /home/sheng/paper6/scripts/sheng_env.sh
python /home/sheng/paper6/scripts/audit_liga_run.py \
 --manifest /home/sheng/paper6/data/runs/liga-clean-seed17-003.json \
 --source-manifest /home/sheng/paper6/data/provenance/server-source-manifest-sensor-spawn.json \
 --preparation /home/sheng/paper6/data/kitti-preparation-001.json \
 --output /home/sheng/paper6/data/runs/liga-clean-audit-003.json
```

Full Car3D AP_R40 IoU0.7:86.855334/67.723812/62.028676% (Easy/Moderate/Hard).
The independent audit also recomputes Pedestrian/Cyclist AP and verifies22388
predictions/52 empty frames. See [full baseline record](liga-baseline.md).

## First real exploratory codec epoch

Fixed protocol `experiments/geometry-link/tuning.md`, SHA
59623ef0aed2c9e751bcd6ee1517777efd16b084366a19121d3af198e7da996f.
Original3712 training frames yield3340 codec-training/372 holdout frames by
fixed hash rule. Original detector pretraining included the holdout; this is
an exploratory codec-only fold, not an independent detector-pretraining fold.
The main3769 validation IDs are excluded.

Run `data/runs/tuning-student-uniform-seed17-002.json` has executed over293
actual optimizer updates. Uniform allocation, student and all auxiliary heads,
AWGN random[-5,20]dB, width4, AdamW0.001, one full epoch. Original parameters
and BN statistics frozen; native imitation NormalizeLayer running buffers still
update during their training-only loss and global_step counts forward batches.
These are explicit checkpoint-audit exceptions, not trainable receiver weights.
Loss and preclip gradient norm must remain finite; clipping10 is active.
No holdout AP is reported until the epoch checkpoint and full evaluation finish.

Log: `/home/sheng/paper6/logs/tuning-student-uniform-seed17-002.log`.
Output: `/mnt/d/paper6/runs/liga/home_sheng_paper6_configs_tuning/student_uniform_seed17_epoch1.tune_uniform_seed17_002`.
Four deployed trees match frozen `*-source-manifest-tuning-002.json`.
Attempt001 failed on Python3.10 `collections.Iterable` before any update.
Its raw evidence is preserved; `modernize_liga.py` now uses collections.abc.
`patch_training_checks_liga.py` activates configured clipping/nonfinite rejection.
Later read-only scalar audit scripts have separate identities and do not alter
the already initialized trainer. Keep attempt002's original source manifest.
