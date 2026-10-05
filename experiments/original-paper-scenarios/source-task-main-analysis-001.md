# Complete source-only detector comparison on KITTI val3769

All twelve JPEG/JPEG2000 x compression10/30/50 x two detector endpoints finish
24 native commands with returncode0. Native closure verifies shared received
pixels, frozen checkpoints, preprocessing and official AP replay. Independent
local proof checks every45,228 frame record,197,185 detection boxes and45,303
artifacts, including24 metric/evaluator files. No wireless channel is present.

Car3D AP R40, Moderate, IoU0.7, percent:

| Source | Nominal CR | Actual CR | PSNR dB | Stereo R-CNN | LIGA |
| --- | ---: | ---: | ---: | ---: | ---: |
| JPEG | 10 | 11.019 | 29.915 | 33.883 | 65.019 |
| JPEG2000 | 10 | 10.004 | 36.093 | 32.188 | 62.650 |
| JPEG | 30 | 33.205 | 27.993 | 29.947 | 56.954 |
| JPEG2000 | 30 | 30.018 | 30.438 | 28.557 | 51.681 |
| JPEG | 50 | 54.666 | 26.803 | 20.797 | 44.571 |
| JPEG2000 | 50 | 50.037 | 28.652 | 25.228 | 45.458 |

At nominal10/30, JPEG2000 has higher pooled RGB PSNR but lower3D AP for both
detectors. These observations establish a discrepancy between this image
quality measure and task ranking; they do not isolate stereo-depth errors or
establish a causal geometric mechanism. Actual compression/bytes differ:
JPEG2000 spends more bytes at each nominal setting. No exact rate-matching,
continuous interpolation guarantee, PHY benefit or novel-method AP claim.

![Actual source bytes, RGB quality and full-split AP](/Users/chen/Documents/ChatGPT/paper6/to_human/source-quality-task-main-001.png)

The dependency driver's original artifact-count assertion omits24 already
required metric files. Preserve failed001. Prelock9c8d93a and isolatedc862173
repair only count/mapping validation; no detector/audit/native closure rerun.
All45303 artifacts then transfer and verify. Unchanged SRCNNformal launcher
starts once, PID282374, currentrate10 actually verified running. All111360
training updates and full native/local acceptance remain pending.

Evidence: source-cache-inference-main-001-closure.json and complete local
verification.json, original-source-cache-main-001-local-verification.json,
source-quality-main-001-local-verification.json, source-task-main-results-001.csv.
