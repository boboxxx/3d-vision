# SRCNN adaptation: native engineering fully closed

The fixed three-rate KITTI adaptation now has executable training and complete
independent all-update auditing. Protocol2a610aa preceded five-file
implementationd74ac31. All four original CPU families passed on both hosts:
direct original-MAT float32 convolution, central-patch/full-image agreement,
six gradients/loss normalization, and complete20epoch sampling coverage.
Maximum absolute convolution error against the independent SciPy reference
is1.430511474609375e-6 on both hosts, below the fixed2e-5 gate.

First engineering001 is a retained failure. Actual cr10 performed6updates and
wrote final weights, then the training-input barrier rejected its attempted
checkpoint hash read. cr30/cr50 did not start; no successful001 audit or closure
exists. Failed manifests, logs, raw rows and written weights are retained and
locally checked. They are never the initialization or selected result of002.

Checkpoint repair002 was separately prelocked. Only serialization changed:
write a torch BytesIO checkpoint, hash the complete memory bytes, then write
them once. Training still cannot read any checkpoint/validation/label/calibration
input. Both original and repaired code files remain pinned. Five CPU families
now pass on both hosts, including real serialized-state read-back from memory
and retained forbidden weight-file access. Six initial float32 parameter
identities are exactly the same as001 and the original converted author model.

Unique native controller268663 completed three rates, each6updates and its
independent actual-PNG/patch/Adam audit:18updates/576training patches total.
All4unique views and all24view visits per rate use000000/000003 only, with
identical view orders, actual patch coordinates, source pixels and target tensor
hashes across rates. Degraded inputs differ only through their declared rate.
Six parameters/gradients/Adam states per rate are finite and have exact step6.
Author MAT and all old/new source files remain unchanged. TF32 and mixed
precision are off. Peak reserved memory is243269632bytes (232MiB) per rate,
with the required measured2GiB physical margin.

Actual process exit, all six commands, native21-artifact closure and independent
local21-artifact verification pass. Local verification checks all18rows,
three actual stored weight/Adam checkpoints and exact shared exposure; the
complete actual PNG/preprocessed-patch reconstruction happened on sheng.
No local GPU/backpropagation replay, validation image, detector, radio or AP.

| Nominal rate | Step1 loss | Step6 loss |
|---|---:|---:|
| 10 | 0.0010717462 | 0.0012125373 |
| 30 | 0.0036746489 | 0.0039314697 |
| 50 | 0.0044859028 | 0.0055895797 |

These are different randomly sampled patch batches, not a fixed validation
comparison or convergence finding. No hyperparameters or budget change follows
from them. All engineering weights are discarded. Formal002 remains exactly
37120updates/rate over public train3712, final-only weights, and cannot launch
until the existing twelve complete JPEG/JP2 detector endpoints have actual
terminal and local verification. Formal adaptation, float received-cache gate,
all3769source/detector results and the full original matrix remain unfinished.

Evidence: `srcnn-kitti-engineering-seed17-002-{closure,local-verification}.json`,
all three native audits/checkpoints/raw logs, and the retained001 failure proof.
