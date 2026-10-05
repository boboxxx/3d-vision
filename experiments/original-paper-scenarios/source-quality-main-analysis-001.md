# Complete main source-quality evidence

All six JPEG/JPEG2000 conditions over3769 fixed validation stereo pairs,
45228 RGB8 views, finish native measurement, independent fresh-pixel integer
MSE/SSIM audit and actual-terminal closure. Three controller commands exit0;
PID270132 is gone. Every transferred row/source/ROI/cache lineage is locally
verified and all pooled aggregates independently recomputed.

PSNR is computed from pooled RGB MSE, not mean per-image dB. SSIM uses the
frozen11x11 Gaussian1.5 reflect/population/channel policy. Each condition has
7538 full-image views and6868 defined ROI views;670 empty ROI views remain
undefined, never scored as perfect. Public original ROI masks are fixed.

| Condition | Full PSNR dB | Full SSIM | ROI PSNR dB | ROI SSIM |
| --- | ---: | ---: | ---: | ---: |
| jpeg-cr10 | 29.914626 | 0.905252 | 29.021155 | 0.922082 |
| jpeg-cr30 | 27.993102 | 0.855858 | 27.088187 | 0.875732 |
| jpeg-cr50 | 26.803178 | 0.809286 | 25.869058 | 0.829575 |
| jpeg2000-cr10 | 36.092790 | 0.940766 | 36.332110 | 0.953208 |
| jpeg2000-cr30 | 30.438363 | 0.864630 | 30.390517 | 0.894336 |
| jpeg2000-cr50 | 28.652154 | 0.829412 | 28.344674 | 0.861706 |

These are source-only image quality metrics; no wireless symbols/energy or
AP claim follows. JPEG actual ratios11.01883053/33.20497649/54.66592980 and
JPEG200010.00423422/30.01799869/50.03667171 are reported as measured, rather
than exact resource matching. Main detector comparisons remain in progress.

Evidence: [native closure](/Users/chen/Documents/ChatGPT/paper6/data/provenance/source-quality-main-001-closure.json),
[all45228-record local proof](/Users/chen/Documents/ChatGPT/paper6/data/provenance/source-quality-main-001-local-verification.json),
[actual controller](/Users/chen/Documents/ChatGPT/paper6/data/runs/source-quality-main-001-cycle.json).
Raw pixel replay happened natively; local verification covers saved rows,
identities and pooled arithmetic, without claiming local raw-pixel replay.
