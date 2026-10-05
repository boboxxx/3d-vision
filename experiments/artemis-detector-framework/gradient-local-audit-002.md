# Preserve secondary bit-equality failure and verify original backward gate

Verifier001 added a bit-for-bit post-NMS comparison to a no_grad forward;
gradient-protocol001 never required this. Inputs/states remain identical and
both outputs are finite; the observed box maxima are 3.06368e-5/7.17640e-5,
score maxima 8.22544e-6/2.07126e-5, classes/counts unchanged. Preserve the full
failure and diagnostic. The cause of output rounding differences is unresolved.

Verifier002 checks the original locked backward requirements unchanged:
actual loss, independent loss-side GT conversion, every saved gradient/target,
finite/nonzero both-image gradients and all484 states unchanged, sensor-only
entry, no optimizer. Exact source input comparisons remain. Record every
post-NMS numerical delta without claiming output bit equality or a numerical
gradient oracle. This closes autograd evidence only, not forward numerical
equivalence, optimized codec training or scientific geometry benefit.
