# Geometry-aware stereo semantic communication

Research project targeting IEEE ICC 2027. Original reference: Cao et al.,
*Task-Oriented Semantic Communication for Stereo-Vision 3D Object Detection*,
TCOM 2025, [arXiv:2502.12735](https://arxiv.org/abs/2502.12735).

Current work tests geometry-dependent power allocation and direct latent
communication using the complete official LIGA-Stereo detector. This is a
proposed method; no supported geometry-link improvement or original-paper
reproduction claim exists yet.
The original paper's Stereo-RCNN/image-reconstruction pipeline remains a
separate required baseline, not replaced by LIGA-Stereo.

See [research plan](experiments/geometry-link/protocol.md),
[source audit](literature/survey.md), and [current findings](findings.md).

The local workspace holds source and evidence; substantive experiments run on
the user-designated sheng and Artemis servers. Local checks verify arithmetic
and transferred evidence. See the [latest progress report](to_human/progress-2026-10-04.md).

The declared original RGB reconstruction variant completed all83 training
epochs and277220 updates with complete state/optimizer audits. Six internal
372-frame detector endpoints using shared received RGB are closed; their mean
resource is200065 complex symbols/frame. Undisclosed author settings prevent
an exact original implementation claim. See [trained baseline evidence](experiments/original-detector-integration/final-analysis-001.md).

The direct-latent matched channel-adaptation comparison completed3340 updates
per arm and four internal372-frame endpoints at62400 symbols/frame. AWGN10
Moderate Car AP_R40 at IoU.7 is17.4029% after identity continuation and24.5328%
after noisy-channel adaptation. This single-seed training-set diagnostic
supports channel adaptation under equal update/resource budgets; it is not
main-validation or geometry-allocation evidence. See [F7 analysis](experiments/geometry-link/F7/analysis.md).

The frozen geometry-risk pilot completed64 training frames/8320 native loss
passes. Complete independent native and five-model statistical audits pass,
including local independent recomputation. Geometry's positive-damage rank
correlation is.0446 with a95% frame-bootstrap interval spanning zero; gradient
strongly orders fluctuation variance(.8834). This does not support a geometry
mean-risk allocation head. Two audit-reference failures are preserved; native
observations and sources did not change. See [full64 analysis](experiments/geometry-risk/pilot-analysis-001.md).

The fresh8 paired-noise follow-up also closed all8208 native passes and server/
local independent audits. Frozen geometry predicts fluctuation variance(.4512)
but supplies no positive signed-mean damage evidence(−.0598, interval spanning
zero); gradient variance ordering is.9601, with substantial linear residuals.
See [paired diagnosis](experiments/geometry-risk/antithetic-analysis-001.md).
The next [F8 protocol](experiments/geometry-link/F8-matched-encoder-adaptation.md)
locks equal-update codec-only versus joint student/codec adaptation. Both six-step
native engineering arms and local verification passed; formalPID167748 is running,
with no F8 AP endpoint yet. Main validation, allocation, fading and multi-seed evidence
remain required.

On sheng, the complete author model and new link now pass full-resolution GPU
engineering forward/backward checks. All 484 released detector weights match;
the original evaluator passes independent synthetic and analytic IoU fixtures.
All five KITTI archives and 7481 files per training folder are verified.
The branch-matched Stereo-RCNN clean evaluation completed all 3769 frames:
Car 3D AP_R40 at IoU 0.7 is 53.122/34.075/27.854% (Easy/Moderate/Hard).
A completed compatibility run with mismatched branch weights is retained separately.
LIGA train/val infos are verified; full clean evaluation run003 completed all
3769 frames and independent prediction/GT/AP audit passed. Car 3D AP_R40 at
IoU 0.7 is 86.855/67.724/62.029% (Easy/Moderate/Hard). These are separate
clean detector references, not a communication-method gain. See the actual
deployment and remaining gates in [server record](docs/server.md).

The original downstream Stereo-RCNN is also deployed on sheng with author weights.
Independent operators and complete demo 3D solve pass; the headless result matches
the actual author demo exactly. See [baseline record](docs/stereo-baseline.md).
This recovers the clean downstream detector, not the original communication codec.

## Development

```bash
python3 -m unittest discover -s tests -v
python3 scripts/patch_liga.py --root third_party/LIGA-Stereo
python3 scripts/modernize_liga.py --root third_party/LIGA-Stereo \
  --mmdet third_party/mmdetection_kitti
python3 scripts/patch_training_checks_liga.py --root third_party/LIGA-Stereo
```

The patch supports optional communication boundaries before or after early
stereo cost processing, plus a separately configured RGB reconstruction link. Every image feature needed downstream crosses this boundary.
When disabled, the upstream detector prediction path is preserved. Runtime
modernization also removes training-only teacher computation and optional GT
depth diagnostics from sensor-only inference. Server setup uses
`source scripts/sheng_env.sh`, including NVIDIA's supported Numba CUDA binding.

The current optional student sender measures 12.48 G dense FLOPs lower bound
and 15.99 ms transmitter stream time, retaining the complete original receiver.
Its original feature teacher runs only during training. Raw-cost posterior logits
use a nonlinear head so left evidence can affect depth probabilities.
See [compute audit](docs/compute-audit.md), including earlier unfavorable costs;
embedded deployment benefit is not established. Main-validation communication
results and geometry allocation remain open. Optional task
sensitivity uses native detection gradients as training-only supervision and
predicted importance at inference; full GPU checks pass. A locked one-epoch
learning check uses 3340/372 frames drawn only from the original training set;
see [exploratory protocol](experiments/geometry-link/tuning.md). Its holdout
isolates new codec training, not the released detector's pretraining.
Earlier failed student/cost-boundary experiments remain in the
[diagnostic protocol](experiments/geometry-link/diagnostics.md) and findings.
Feature matching alone did not establish useful task transport; native task
adaptation and matched channel exposure were subsequently completed. Allocation
still requires predictive risk evidence, actual signaling and equal budgets.

Artemis has verified identical complete KITTI data and RTX PRO6000 Blackwell
hardware/runtime checks. Native sm120 detector operators compile, but their
GPU probe11423980 is pendingQOSGrpGRES; full detector compatibility is open.
Its actual CPU LDPC/QAM/APP/BP chain and counted receiver framing are audited,
including erasures and cross-platform JP2 decoder differences. ECSIC actual
entropy streams, learned quantizer calibration, original full scenario grid,
fading, multiple seeds and the manuscript remain required work.
