# Q1: flexible mass fixes one source error, not wireless transport

All23,296 matched synthetic Q0/Q1 cells from both hosts pass independent
spatial-CDF and scalar projection replay;944 actual fit candidates and the
previous10,752 Q0 records also pass. At the same two complex symbols/energy2,
Q1's optimal two-node law retains unequal modal probability, sacrificing the
third location. These are synthetic distribution distances, not KITTI depth
truth or localization errors.

| Fixed source | Q0 intrinsic W1, m | Q1 intrinsic W1, m | Q0 AWGN10 full W1, m | Q1 AWGN10 full W1, m |
| --- | ---: | ---: | ---: | ---: |
| Uniform | 4.8000 | 7.2000 | 6.8908 | 8.9975 |
| Two modes 50/50 | 8.0000 | 0.2000 | 11.9233 | 8.7906 |
| Two modes 35/65 | 0.9187 | 0.2000 | 6.1980 | 9.0425 |
| Far minority 5% | 2.4491 | 0.2000 | 6.6028 | 9.7127 |
| Far minority 20% | 6.4333 | 0.2000 | 10.5827 | 9.5250 |
| Three modes | 0.2000 | 8.0333 | 5.4893 | 11.1337 |

All13 sources and identity/AWGN/Rayleigh6/10/18 records remain in the
complete artifacts; this illustrative table does not select another primary
metric. The intrinsic improvement for separated binary modes is real, but
mass error costs distance between the modes. Noise on the explicit mass can
outweigh improved source approximation, including the5% minority case.

At the predeclared two-mode ±1e-8 perturbation, source-law W1 is9.6e-7m;
Q0's carrier distance1.3202 reflects a median crossing a zero-density gap,
whereas Q1's carrier distance3.6573e-8 is continuous in this case. The
three-mode ±.01 probe changes Q1's selected split and gives a large carrier
change; the smaller finite-width histogram probes do not exhibit an
infinitesimal jump. Do not claim such a jump from this evidence. Smooth carrier
gradients were checked, but optimal fitting remains hard/non-differentiable.

The global scalar quantizer is verified against exhaustive240,000 midpoint
sample partitions to the declared W1 approximation bound. Independent complete
CDF replay has maximum histogram error1.03e-13m and cross-host physical error
2.84e-14. Neither standard optimal quantization nor sphere coding is claimed
as novel. Q1 is not promoted to main detector training. Next determine whether
native sender laws preserve useful task information, with matched generic
representation/adaptation/compute/resource controls; source approximation alone
is insufficient.

Evidence: [complete proof](/Users/chen/Documents/ChatGPT/paper6/data/provenance/depth-particles-Q1-complete-local-verification-001.json),
[locked protocol](/Users/chen/Documents/ChatGPT/paper6/experiments/depth-particles/protocol-002.md),
[all local arrays](/Users/chen/Documents/ChatGPT/paper6/data/engineering/depth-particles-Q1-local-CPU-001/complete_arrays.npz).
