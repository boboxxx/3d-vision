# Native sender posterior: lower source error still loses under noise

Attempt002 covers every99,840 fixed-grid cell of two training frames/four
views, including1,641 invalid cells with charged dummy transmissions. Native
independent replay passes all98,199 valid laws,4,713,552 candidate partition
costs,599,040 method/condition cells and1,198,080 attempted complex uses.
Each method has the same two complex symbols/energy2 per cell, received-only
decoder and common PCG2806 AWGN10 draws. Complete local spatial-CDF/direct-cost replay also passes all five transferred
artifacts; the native and local evidence set is now closed.

| Representation | Identity mean W1, m | AWGN10 mean W1, m | AWGN10 p95 W1, m |
| --- | ---: | ---: | ---: |
| Q0: three equal-mass quantile atoms | 4.00037 | 12.05784 | 24.01050 |
| Q1: two atoms and explicit probability | 3.23188 | 12.99819 | 28.55624 |
| Q2: same two-node law with mean coordinate | 3.23188 | 13.80381 | 28.91236 |

Q1 improves source approximation by0.76850m on these uncalibrated cached laws,
but AWGN10 full-law distortion increases0.94035m versus Q0. Q2 preserves the
same source law and its received-only probability recovery is well-defined
at coincident endpoints; it increases noisy distortion a further0.80562m.
Every view has positive Q2−Q1 mean difference:0.75528,0.82066,0.81143,0.83466m.
Thus replacing probability with the mean does not solve this carrier's noise
problem. The pointwise coupling/anchor bounds pass, but are loose upper bounds
and do not imply competitive average transport performance.

The original cached source is the frozen F6b535-state student's cosine
stereo feature posterior, with48 masked disparity atoms and temperature.1.
It is not the author484-state detector's native depth prediction or a calibrated
posterior. The depth pushforward uses the original float32 camera fB and
z=fB/d with fixed public[0,128]m coding bounds. No image, feature, gradient,
GT or label array is deserialized. Full original archives are hashed before
and after, and independently re-read probabilities/metadata match every
sanitized output. Clean validity affects evaluation only, with all invalid
symbols transmitted and decoded normally. No mask/address oracle is supplied.

Attempt001 remains fully retained and is not accepted. Its direct candidate
cost audit caught an empty-cluster prefix-formula error (41.51349m direct vs
124.54047m recorded for one candidate). Protocol004 prelocked zero costs for
empty segments without changing objective, input, source exposure, resources,
noise or decoder. Attempt002 then passes direct weighted-absolute-cost and
spatial-CDF replay. The repair is an implementation correction, not a changed
scientific hypothesis. A full array comparison finds only candidate_W1 changed; every selected law
and every transmitted/received/decoded record is bitwise identical to attempt001.
Attempt001 remains unaccepted despite that comparison.

These four cached train views are an exploratory representation diagnostic,
not full KITTI evaluation. The distance is between probability laws, not
measured true depth/localization error. No new-method advantage, calibrated
uncertainty, detection/task preservation, differentiable fitting or novelty
is established. Do not promote this hard two-node/anchor family to main
training on source approximation alone. The next critical evidence is native
cost-feature/depth information usable by the task decoder under matched
appearance, generic latent, adaptation, compute, exposure and resource controls.
Full original wireless conditions and multiseeds remain required.

Evidence: [native full proof](/Users/chen/Documents/ChatGPT/paper6/data/provenance/depth-particles-native-Q2-native-verification-002.json),
[local full proof](/Users/chen/Documents/ChatGPT/paper6/data/provenance/depth-particles-native-Q2-local-verification-002.json),
[original prelock](/Users/chen/Documents/ChatGPT/paper6/experiments/depth-particles/protocol-003.md),
[retained failure and repair prelock](/Users/chen/Documents/ChatGPT/paper6/experiments/depth-particles/protocol-004.md).
