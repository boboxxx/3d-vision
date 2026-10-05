# Complete identity audit with unresolved depth criterion

Verifier002 reaches the originally locked FP64 mean criterion and fails.
Retain its full code/log and do not relax 3e-5 to the observed maximum.
Diagnostic001 measures complete two-frame discrepancies 4.33294e-5 and
8.13361e-5; native code uses FP32 softmax and reduction. This suggests a
precision issue but does not establish its cause.

Audit003 retains every source/state/array/pixel/calibration/box check and the
same complete FP64 recomputation. It records the original criterion for each
frame rather than aborting at the first mismatch, enabling complete local
identity coverage. Its output is an audit with explicitly unresolved strict
depth acceptance, never a passed complete numerical proof. No native artifact,
model or saved value is changed. A separate numerical investigation remains.
