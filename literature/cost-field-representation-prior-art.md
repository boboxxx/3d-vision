# Cost-field representation boundary, read2026-10-05

Primary author abstract records only, not full-paper reproduction:

- [CFNet, CVPR2021](https://arxiv.org/abs/2104.04314) explicitly uses fused
  low-resolution cost volumes and variance-driven adaptive disparity search.
  Uncertainty-guided cost-volume narrowing is established prior art.
- [Cascade Cost Volume, CVPR2020](https://arxiv.org/abs/1912.06378) narrows
  depth/disparity intervals across feature-pyramid stages for memory/runtime.
  Adaptive depth support and compact cost-volume computation alone are not new.

Other verified local notes already cover Wasserstein multimodal disparity
prediction/3D detection and power-constrained geometric analog mappings.
Therefore neither equal-mass barycenters, native uncertainty, phase encoding,
cost compression nor direct task decoding can individually support novelty.

Protocolcost-field001 asks a narrower operational question: at the same noisy
complex-symbol budget, initial detector/pretraining and codec capacity, does
native metric-depth-conditioned feature integration improve3D task performance
over a fully learned generic integration? The codec transmits coordinates and
conditional multichannel appearance/cost values; receive-side native depth,
voxel and detection fields are derived entirely from the corrupted packet.
Geometry organization is separately controlled by spatially shuffled priors.
No empirical answer, absence-of-prior-art or full novelty claim is established.
Full-text comparisons and actual multi-seed matched task results remain required.
