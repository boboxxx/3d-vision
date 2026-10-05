# Available experimental results

Snapshot date: 2026-10-05T16:49:58.416388+00:00. These tables preserve recorded values; incomplete
detection evaluations are not filled from synthetic or exploratory results.

## Full-split source-compression results

All 3,769 KITTI validation stereo pairs; no wireless channel. AP is Car 3D
Moderate AP_R40 at IoU 0.7, in percent. PSNR uses pooled RGB MSE.
Actual source rates differ at equal nominal compression ratios.

| Codec | Nominal CR | Actual CR | Bytes/pair | PSNR (dB) | Stereo R-CNN AP | LIGA AP |
|---|---:|---:|---:|---:|---:|---:|
| JPEG | 10 | 11.019 | 252,943 | 29.915 | 33.883 | 65.019 |
| JPEG | 30 | 33.205 | 83,937 | 27.993 | 29.947 | 56.954 |
| JPEG | 50 | 54.666 | 50,985 | 26.803 | 20.797 | 44.571 |
| JPEG2000 | 10 | 10.004 | 278,596 | 36.093 | 32.188 | 62.650 |
| JPEG2000 | 30 | 30.018 | 92,849 | 30.438 | 28.557 | 51.681 |
| JPEG2000 | 50 | 50.037 | 55,702 | 28.652 | 25.228 | 45.458 |

Source: [recorded CSV](../experiments/original-paper-scenarios/source-task-main-results-001.csv).
JPEG2000 has higher PSNR but lower AP for both detectors at nominal CRs 10 and 30.
This is a ranking observation, not a matched-rate communication-method gain.

## Synthetic position receiver at 10 dB

Uniform prior over 6,545 inclusive ordered triples on a 33-point grid.
Each trial transmits three unit-energy complex position symbols. Coordinate MSE
is scaled to m² by 57.6²; it is not a detected object-depth error.

| Receiver | AWGN MSE (m²) | Perfect-CSI Rayleigh MSE (m²) |
|---|---:|---:|
| Clipped phase | 16.1234 | 69.3591 |
| Independent uniform-slot posterior | 15.4297 | 46.4875 |
| Weighted isotonic projection | 14.3532 | 40.8337 |
| Ordered fading-blind posterior | 12.9310 | 31.5355 |
| Ordered CSI-conditioned posterior | 12.9310 | 26.5377 |

| Scalar-field reconstruction | AWGN MSE | Rayleigh MSE |
|---|---:|---:|
| Posterior basis expectation | 0.00059813 | 0.00110864 |
| Basis at posterior-mean coordinates | 0.00060495 | 0.00116890 |

The field uses fixed coefficients (0.8, -0.3, 0.5), nine queries, and
normalized-coordinate bandwidth 1/3. The independent posterior uses different
slot marginals, so its gap does not isolate the effect of joint ordering.

Evidence: [two-host closure](../data/provenance/ordered-receiver-two-host-closure-017.json),
[local report](../data/engineering/ordered-receiver-local-016.json), and
[analysis and limitations](../experiments/cost-field/ordered-receiver-analysis-017.md).

## Current conditional codec

- Seed 17, four codec arms, three epochs per arm, 3,712 training pairs.
- 12 epoch checkpoints retained and checked; weights are not part of this export.
- Full two-host training verification remains incomplete in the retention record.
- The primary full-validation AP comparison remains pending in the current record.
- [Checkpoint evidence](../data/provenance/cost-field-all12-checkpoint-retention-014.json).
- [Native training verification](../data/provenance/cost-field-full-seed17-004-native-audit-007.json).
- [Fixed evaluation protocol](../experiments/cost-field/final-validation-original-grid-008.md).

## Earlier experiments and plots

- [Historical experiment findings](../findings.md).
- [Full experiment log](../research-log.md).
- [Figure gallery and progress reports](../to_human/).
- [Reference-inspired RGB baseline analysis](../experiments/original-detector-integration/final-analysis-001.md).
- [Matched channel-adaptation analysis](../experiments/geometry-link/F7/analysis.md).
- [Geometry-risk pilot](../experiments/geometry-risk/pilot-analysis-001.md) and
  [paired-noise follow-up](../experiments/geometry-risk/antithetic-analysis-001.md).

Historical diagnostic metrics use different frame splits and are not pooled
with the full-validation source-compression results above. Reports referring
external raw files can be traced through the external-artifact index.
