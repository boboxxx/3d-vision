# F3 feature distortion diagnosis, locked after the identity AP result

Identity-codec Car Moderate3D AP_R40 is0.0398054% (372-frame independent
audit), so noise removal alone is insufficient. Observe unchanged finalF3
weights on the same ordered372 holdout frames, identity channel, uniform power,
seed17 and native sensor-only inference. This performs zero training updates.

A passive forward hook records raw cost/appearance feature shapes, pooled
shapes, element counts, NMSE, MSE, cosine, RMS, mean and negative-value fraction
between codec input and interpolated decoder output. Sum reductions are chunked
float64 to avoid duplicating the entire cost volume in float64. Nonfinite
features fail. Undefined zero-norm NMSE/cosine are null, never silently floored.
No image, GT label, LiDAR or predicted box is consulted by the hook.

Also measure input against plain average-pool/interpolate reconstruction at
the same spatial/depth strides. This is a diagnostic reference, not an optimal
reconstruction, irrecoverable-information bound or AP ceiling. It must not be
called an oracle. Compare feature distortion before proposing new capacity,
pooling or reconstruction-pretraining changes. Do not infer a unique cause
from aggregate feature errors alone.

The full unchanged native test runner still writes predictions/AP and372
communication records. Preserve and independently audit those artifacts,
ordered feature-record coverage and source/checkpoint identities. Do not
choose a better AP rerun: F3 and identity results stay their first locked runs.
