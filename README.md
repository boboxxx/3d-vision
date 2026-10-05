# Stereo 3D Vision Semantic Communication

Code and experimental evidence for a research project targeting IEEE ICC 2027.
The reference problem is *Task-Oriented Semantic Communication for Stereo-Vision
3D Object Detection* ([Cao et al., TCOM 2025](https://arxiv.org/abs/2502.12735)).
This repository records the current method, earlier experiments, and completed
results. It is a research snapshot rather than a completed reproduction of the
reference paper.

## Current method

The conditional depth-field codec connects a frozen LIGA-Stereo front end to a
frozen detection head. Each pooled spatial location is represented by three
equal-mass depth barycenters, their conditional feature vectors, and an
appearance vector. The packet consumes **19 complex symbols per location**
(**29,640 per stereo frame**) with explicit energy accounting. Four arms compare
the aligned guide, no guide, shifted guide, and a dense depth bottleneck.

The separate ordered-posterior receiver uses received symbols and channel state
information to average the reconstruction basis. Its completed synthetic
experiments use a declared discrete prior; they do not measure detector AP.

## Results available in this snapshot

- JPEG/JPEG2000 source-compression experiments on **all 3,769 KITTI validation
  stereo pairs**, evaluated with Stereo R-CNN and LIGA-Stereo.
- Ordered-receiver experiments covering **20 channel/SNR conditions** and
  **1,310,720 synthetic triples per host**, with retained-evidence verification
  reports for local and sheng runs.
- Completed conditional-codec training evidence for seed 17: **12 retained epoch
  checkpoints** across four arms. Checkpoint files are external; full-split
  learned-codec detection AP is **pending** in the current evidence record.
- Earlier clean-detector, communication-adaptation, geometry-risk, and
  engineering experiments, with their original limitations preserved.

Start with [the result tables and evidence index](results/README.md),
[the method description](docs/cost-field-method-004-current.md), and
[the manuscript source](paper/icc2027.tex). Figures are in [to_human](to_human/).

## Code map

| Component | Path |
|---|---|
| Current frozen conditional codec and training | `experiments/cost-field/code_004/` |
| Full-split inference and evaluation | `experiments/cost-field/validation_code_008/` |
| Ordered posterior and synthetic checks | `experiments/cost-field/ordered-receiver-code-015/` |
| Received-only detector interface | `experiments/cost-field/forward_with_cut.py` |
| Original-paper source/channel scenarios | `experiments/original-paper-scenarios/` |
| Reference-inspired RGB implementation | `reproduction/cao2025/` |
| Shared communication modules | `src/geocomm/` |
| Detector compatibility, preparation and audits | `scripts/` |
| Configurations and local regression tests | `configs/`, `tests/` |
| Results and provenance | `data/`, `results/`, `to_human/` |

## Environment and reproduction

The lightweight package requires Python, PyTorch, NumPy and PyYAML:

```bash
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
```

Full detector training additionally requires the pinned upstream repositories,
CUDA operators, author detector weights, and a separately acquired KITTI dataset.
See [the server/runtime record](docs/server.md),
[the stereo baseline record](docs/stereo-baseline.md), and
[the full-training protocol](experiments/cost-field/protocol-004-full-training.md).
The recorded GPU training harness checks exact runtime, input hashes and output
identities. It must be configured for a new machine before execution; historical
server paths in evidence files are provenance, not portable installation paths.
No training or evaluation job is launched by this repository upload.

## Snapshot contents

This export includes code, configurations, result tables, prediction text,
reports, plots, literature notes, and the manuscript. It excludes raw KITTI
sensor files, third-party checkouts, model weights, binary intermediate payloads,
TensorBoard archives, and large per-step/per-frame archives above 5 MiB.

[The snapshot record](results/snapshot.json) and
[file manifest](results/files.sha256.jsonl) identify the exported files.
[The external-artifact index](results/external-artifacts.jsonl) lists omitted
tracked artifacts by path, size, reason, and SHA-256. Those artifacts remain in
the experiment workspace; a summary report is not a substitute for replaying an
omitted raw archive. Untracked live audit outputs were not exported.

The original workspace README is retained as
[historical context](docs/workspace-readme-at-export.md). Earlier exploratory
results should be interpreted according to their original protocols. The
included third-party snippets retain their existing attribution and licenses;
full upstream repositories and pretrained weights are obtained separately.
