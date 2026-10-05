# Native geometry engineering auditor repair 001

2026-10-04, after completed260-pass engineering, before repaired audit. The
native experiment and detached controller exited successfully; the independent
CPU auditor failed comparing raw-label-derived GT yaw with actual native GT.
Firstframe actual+3.092389 versus independently reconstructed-3.190796 differs
by2pi. Other seven box entries matched at the existing2e-6 tolerance.

Frozen `stereo_dataset_template.py` uses its test-mode data_augmentor.forward;
`stereo_data_augmentor.py:181` always canonicalizes gt_boxes[:,6] using
limit_period(offset=.5,period=2pi), also for deterministic test crop. The
independent auditor omitted this documented source transformation. Preserve
its initial failure/code SHA. Fix independent expected yaw only by explicit
float32 angle-floor(angle/(2pi)+.5)*(2pi); all GT comparison tolerances, input
hashes/calibration/geometry/noise checks remain unchanged. No native experiment,
model, labels, saved fixtures or frozen source change; no GPU rerun. Audit anew
under the still-unused unique audit output. This repair cannot justify a fit,
AP or allocation claim. A further mismatch must remain a failure.
