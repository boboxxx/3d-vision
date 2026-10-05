# Q2: native cached posterior and barycentric coordinates

Exploratory CPU engineering experiment, locked before execution. Q1's synthetic
source improvement did not ensure noisy transport improvement. Test the same
two-atom law with explicit mass (Q1) or its mean (Q2), alongside three equal-mass
quantiles (Q0), on every cell of four cached train-only sender feature posteriors.
This tests source approximation and coordinates, not task sufficiency or AP.

Inputs are the independently audited geometry-risk-native-engineering-001
fixtures, from the frozen F6b student (535 states), not the author 484-state
LIGA depth head. They contain an uncalibrated cosine epipolar posterior over
48 disparity atoms, input pixels 4,8,...,192, temperature .1 and masked invalid
correspondences. No fitting, GT supervision or new detector execution is used.
Read only left/right_probability, left/right_valid, disparity_bins, P2, P3,
valid_image_shape. Never deserialize images, features, gradients or GT arrays.
Hash the complete original archive before and after reading. Locked inputs:

| Frame | Native archive SHA256 | Bytes |
| --- | --- | ---: |
| 000000 | cbb10d56055ec6c540b5f103d25a904e67d80c06300580fe6ee3104fa927fd17 | 118431696 |
| 000003 | 402f4c91fc9a26100d8e20f14a16b6b9215a65d93d39858d6986ad8299062a58 | 117746968 |

Archives reside in /mnt/d/paper6/runs/geometry-risk-native-engineering-001.
Native provenance is data/provenance/geometry-risk-native-engineering-001-audit.json.
Derive fB=abs(P2[0,3]-P3[0,3]) in their original float32 arithmetic, matching
the frozen Calibration.fu_mul_baseline property. Push each discrete disparity
mass to depth fB/d, reversing axes into increasing depth. Do not replace discrete
atoms by histogram bins. Global public communication bounds [0,128] metres;
assert the whole support is inside without clipping or per-view normalization.

Q0 uses weighted discrete quantiles at 1/6,1/2,5/6. Q1 enumerates all 48
contiguous two-cluster partitions, conditional weighted medians and exact L1
costs. A boundary at a discrete source atom suffices: for fixed ordered centers,
nearest-center assignment gives contiguous clusters; any tied atom can be
assigned wholly to either side without increasing cost. Zero-mass cluster uses
the remaining median. Ties within 1e-11 metres select the smallest first mass.
If both fitted centers coincide, canonicalize the unused mass to .5. Preserve
all candidate splits/objectives and selected indices for complete verification.

Q2 uses the exact same Q1 fitted law, with coordinates [a,mu,b], where
mu=m*a+(1-m)*b. Normalize all depths with the global bounds and use Q0's
unchanged ordered three-coordinate carrier/projection. At the receiver,
m=(b-mu)/(b-a) for distinct endpoints, and .5 for coincident endpoints. No
clean side information or gap floor enters decoding. A coupling gives
W1<=2[m*abs(delta_a)+(1-m)*abs(delta_b)]+abs(delta_mu)
<=sqrt(5)*norm(delta_[a,mu,b]); combined with the anchor inversion bound it
gives W1<=sqrt(10)*128*norm(effective_noise). The bound is looser than Q1's
global bound; no noise advantage is assumed and no novelty is claimed for
scalar quantization or coordinate changes.

Fixed grid 80x312 per view: 99840 cells total. Every cell, including invalid
padding/correspondence cells, transmits two complex symbols with energy2;
invalid sources use public dummy coordinates [0,0,0]. Decode them normally.
Clean validity is used only for evaluation, never selection, allocation or
received gating. Undefined source errors are NaN. No uncharged mask channel.
Identity and AWGN10 are engineering conditions, not the final radio matrix.
PCG64 seed2806, one common four-real standard normal draw per grid cell/view
in frame-major/left-right order, shared by all methods. Real noise variance .05,
Es=1. Six method/condition evaluations charge 1198080 complex uses and energy.
No Rayleigh, GT metric or full task claim in this stage.

Save the complete sanitized source probabilities/calibration/validity, all
candidate fits and every transmitted/received/decoded cell and intrinsic,
channel-only and full W1. Evaluate discrete W1 by quantile-interval overlap.
Independently verify locally using spatial CDF integration over the sorted
union of source/received support; independently enumerate every partition
median cost by direct weighted absolute distances. Check all arrays, PCG
draws, projection, physical budgets, identity inversion, coupling bounds,
source lineage and every aggregate. Retain failures before repairs. Report
both frames/views and pooled valid cells, and Q1/Q2 paired differences.
This diagnostic cannot establish calibrated posterior, true depth error,
teacher preservation, differentiable fitting, novelty or detector benefit.
Matched generic task adapters, multiseeds, full original wireless matrix and
native KITTI 3D evaluation remain necessary before main promotion.
