# Generative Feature Imputing

Jianhao Huang, Qunsong Zeng, Hongyang Du, Kaibin Huang. *Generative Feature
Imputing—A Technique for Error-resilient Semantic Communication*,
arXiv2508.17957v1,2025-08-25.
[Primary manuscript](https://arxiv.org/html/2508.17957v1).
Read2026-10-04: system, power-allocation description and experimental settings.

Spatial packetization and diffusion-based feature recovery accompany
semantic-aware power allocation. CAM estimates packet importance; the allocation
objective is an importance-weighted sum of packet error rates. Experiments use
ImageNet images, pretrained VAE/ResNet components and AWGN/Rayleigh channels.

For this project, packet importance plus channel-aware protection is a relevant
prior and comparator family. Diffusion recovery would add substantial scope;
do not adopt it merely to claim novelty. Our proposed direct3D marginal-benefit
policy must demonstrate task and geometry-specific value beyond a saliency
weight. No unverified venue or numerical gain is asserted here.
