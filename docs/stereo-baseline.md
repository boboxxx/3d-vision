# Original downstream detector recovery

Official repository: https://github.com/HKUST-Aerial-Robotics/Stereo-RCNN

Pinned official Python 3 branch: `4d6dd65049f52c1a5f2b6ad716a7e0da5cb02cb3`.
Master (`63c6ab98b7a5e36c7bcfdec4529804fc940ee900`) uses a different ROI sampling
implementation. The communication paper's exact detector version is unresolved.
The new proposal uses LIGA; the original Stereo-RCNN pipeline remains required.

Author checkpoint, from the repository README's Google Drive ID
`1rZ5AsMms7-oO-VfoNTAmBFOr8O2L0-xt`:

- sheng: `/mnt/d/paper6/checkpoints/stereo-rcnn-author.pth`
- SHA-256: `d2dca213a0092c84a0e2cc96f363755ab0a8cbe00bf49651892ac267050debb4`
- 566 released state tensors match all corresponding model tensors; no missing
  learned weights, unexpected keys or shape mismatches. Modern PyTorch initializes
  104 legacy BN batch counters, which are unused at evaluation.
- Checkpoint metadata reports epoch 21.

`scripts/modernize_stereo_rcnn.py` preserves the full author architecture and
CUDA ROIAlign/ROIPool/NMS kernels. Replaces removed THC APIs with ATen allocation,
current-stream launches and CUDA checks; retains inclusive pixel areas in NMS.
Dense alignment normalizes coordinates by W-1/H-1; explicit align_corners=True
preserves that old grid_sample convention. No torchvision operator replacement.
The model package stays in the checkout, accessed by per-process sys.path; it is
not installed over other projects' `model` packages.

On sheng, in the existing isolated project venv:

```bash
cd /home/sheng/paper6
source scripts/sheng_env.sh
python scripts/modernize_stereo_rcnn.py
(cd third_party/Stereo-RCNN/lib && python setup.py build_ext --inplace)
python scripts/verify_stereo_rcnn.py \
  --checkpoint /mnt/d/paper6/checkpoints/stereo-rcnn-author.pth \
  --output data/engineering/stereo-rcnn-next.json
```

Verification 003 passes independent ROIAlign quadrature and gradient comparisons
in float32/float64, adaptive/fixed sampling, boundaries and malformed ROI cases.
NMS matches an independent inclusive-area implementation for 137 boxes, crossing
the 64-bit mask boundary. Full author demo RGB at 600x1987 runs the 105,891,446
parameter detector, initial box solver, dense disparity enumeration and final
rectification. Six 3D solutions are finite. These six objects are engineering
input, not KITTI AP. Parameter count differs from the communication paper's
complexity table; counting convention/version is unresolved.

`src/geocomm/stereo_baseline.py` adapts the original demo's full postprocess to
return predictions without a GUI or LiDAR visualization. An independent check
executes the actual author's demo with only presentation and legacy assignment
API adaptations. Scores, boxes, dimensions, locations, angles and disparities
match exactly on all six demo solutions. See
`data/engineering/stereo-postprocess-verification-002.json`, which additionally
checks byte-identical KITTI serialization against the actual author writer.
The author converts camera coordinates and subtracts 1.57 from the internal
orientation; preserving these conventions matters to measured 3D AP.

First verification failed because the new validation script accidentally promoted
input to float64. Corrected to the author's in-place float32 mean subtraction;
failure and raw log retained. Verification 002 passes detector forward; 003 adds
the full 3D solve. Cold-pass timings are not latency measurements.

Remaining requirements: original codec recovery, exact
original split/detector version, matched communication and training protocols.
These engineering checks do not establish original-paper reproduction.

## Full validation run

`eval_stereo_rcnn.py` requires all 7481 stereo images, labels and calibration
from verified ZIP archives. Official Stereo-RCNN train/val files match the
fixed OpenPCDet 3712/3769 partition byte for byte. The run loads author weights
and executes the complete 3D solve and dense alignment for every validation
frame. Labels are hashed for identity and read only by the final evaluator.

Initial compatibility run on sheng: `/mnt/d/paper6/runs/stereo-clean-seed17`,
seed 17, PID 8056. This uses branch1 code with the master-released checkpoint
and is not the branch-matched clean baseline.
Log: `/home/sheng/paper6/logs/stereo-clean-seed17.log`. Per-frame prediction
files and hashes are retained. Primary Car 3D AP_R40 uses IoU 0.7; secondary
Car R11/R40 uses IoU 0.5 to expose the original paper's different threshold.
Unchanged evaluator passes a synthetic 100-perfect-car fixture at both
thresholds with an active PyTorch CUDA context. Both runs completed; their results and interpretations appear below. Resume requires an explicit flag,
an identical run identity, and verified saved frame hashes.

```bash
python scripts/eval_stereo_rcnn.py --root /mnt/d/paper6/data/kitti \
  --checkpoint /mnt/d/paper6/checkpoints/stereo-rcnn-author.pth \
  --run-dir /mnt/d/paper6/runs/stereo-clean-seed17 --seed 17
```

The matching branch1 clean run will use `stereo-rcnn-branch1-author.pth` at
`/mnt/d/paper6/runs/stereo-branch1-clean-seed17`. It does not reproduce the
original semantic codec, whose implementation and evaluation details remain
unresolved. Retain existing run directories; do not relaunch a live run.

## Official branch and checkpoint finding

The branch1 README links Google Drive `1rIS43NzTvjRMX9m3UZIG5EvgFzXOVZWX`,
while master links `1rZ5AsMms7-oO-VfoNTAmBFOr8O2L0-xt`. Downloaded the former
to `/mnt/d/paper6/checkpoints/stereo-rcnn-branch1-author.pth`, 846406569 bytes,
epoch 13, 670 model state tensors including all BN counters. SHA-256:
`b7a07a897224f75bc30b2d4c06f39927e92933a8bcaad6e679aff605b2346c14`.

A direct comparison establishes different learned tensors, not merely different
container metadata. See `data/engineering/stereo-branch-checkpoints-001.json`.
The earlier six-demo checks with master weights remain valid operator and
postprocess compatibility evidence. They do not establish a branch-matched
accuracy baseline. The initial full validation run is explicitly classified in
its separate `interpretation.json`; raw metrics/predictions remain untouched.
The correct clean reference must use branch1 source with branch1 weights.

The compatibility run finished all 3769 frames and both AP evaluations;
its raw metrics, run manifest and interpretation are in
`data/engineering/stereo-compatibility-001`. Strict moderate 3D AP_R40 is
9.4557%; this unfavorable compatibility result is retained, not a valid
branch-matched reference or method result. Independent analytic camera IoU
fixtures also pass shifted-height, shifted-position, 90-degree rotation,
containment and disjoint boxes, extending the perfect-detection fixture.

The branch1 checkpoint passes full model verification and independent author
demo/postprocess/writer comparison, with seven finite demo 3D solutions.
Formal branch-matched validation completed all 3769 frames at
`/mnt/d/paper6/runs/stereo-branch1-clean-seed17`; log is
`/home/sheng/paper6/logs/stereo-branch1-clean-seed17.log`.

| Car AP, percent | Easy | Moderate | Hard |
|---|---:|---:|---:|
| 3D R40, IoU 0.7 (primary) | 53.122 | 34.075 | 27.854 |
| BEV R40, IoU 0.7 | 67.562 | 46.594 | 37.675 |
| 3D R11, IoU 0.5 | 85.909 | 72.879 | 57.999 |
| 3D R40, IoU 0.5 | 88.480 | 71.409 | 58.204 |

Local raw metrics, evaluator text and run identity are under
`data/runs/stereo-branch1-clean-seed17`. Independent all-frame artifact audit
`data/runs/stereo-branch1-clean-audit-001.json` passes: exact 3769-frame set,
12542 finite Car predictions, 311 empty frames, zero nonpositive dimensions,
all prediction SHA-256/row counts, fresh label/calibration hashes, source identity
against the frozen run manifest, and run/evaluator metric agreement. It does not
freshly rehash every RGB image; inference-recorded image hashes and archive CRC
identities remain in the run evidence. Per-frame predictions remain on sheng.

These are the clean detector reference under the locked split/evaluator.
They do not reproduce Cao et al.'s communication codec or establish a new
method gain. The original paper's exact detector version and AP sampling
remain unresolved; do not equate the two IoU thresholds.
