# Wasserstein Distances for Stereo Disparity Estimation

Divyansh Garg, Yan Wang, Bharath Hariharan, Mark Campbell, Kilian Q. Weinberger,
Wei-Lun Chao. NeurIPS2020. Verified2026-10-05 from the author
[arXiv record](https://arxiv.org/abs/2007.03085) and the complete
[proceedings PDF](https://papers.nips.cc/paper_files/paper/2020/file/fe7ecc4de28b2c83c016b5c6c2acd826-Paper.pdf), Sections2–3.

The paper predicts offsets alongside discrete probabilities, giving a
distribution over movable disparity supports. It motivates retaining multiple
modes near boundaries instead of reporting their potentially implausible mean.
It trains distribution matching with Wasserstein distances, including
multimodal targets, and evaluates downstream stereo3D detection. Equation12
already expresses one-dimensional W1 as the integral of absolute CDF difference.

Consequently, depth/disparity distributions, multimodality, W1/CDF losses and
their relevance to3D detection are established prior art. A new communication
system would need actual transmitted representation, measured channel cost,
received-only decoding and matched geometry/control evidence. Merely adding
this loss to the existing link cannot substantiate a new contribution.
No reproduction of this author's code or performance was attempted here.
