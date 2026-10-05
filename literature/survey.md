# Primary-source audit (2026-10-03)

## Cao et al. — TCOM 2025

[Paper](https://arxiv.org/html/2502.12735v1), arXiv v1, 18 February 2025.
Authors: Zijian Cao, Hua Zhang, Le Liang, Haotian Wang, Shi Jin, Geoffrey Ye Li.
The paper transmits key-area and global semantics, uses optical flow during
recovery, reconstructs RGB, and uses pretrained Stereo-RCNN. It explicitly
avoids joint training with a specific detector. KITTI is split roughly in half;
reported AP uses IoU 0.5. Exact split, evaluator version, and codec implementation
must be recovered before reproducing numbers. No original communication code
was located yet. This source establishes the reference pipeline, not a novelty
claim for the proposed method.

## Guo et al. — LIGA-Stereo, ICCV 2021

[Paper](https://arxiv.org/abs/2108.08258),
[official code](https://github.com/xy-guo/LIGA-Stereo).
Authors: Xiaoyang Guo, Shaoshuai Shi, Xiaogang Wang, Hongsheng Li.
LiDAR-guided stereo geometry learning and auxiliary 2D supervision. Candidate
complete detector for direct task latent experiments, with stereo-only inputs
at inference and LiDAR supervision in training. Official environment uses
PyTorch 1.6, CUDA 9.2/10.1, modified mmdetection, and spconv. Checked-out revision:
`aee3731a24a0ab1667e633e520cc89be2f135272`.
Actual released config is `configs/stereo/kitti_models/liga.3d-and-bev.yaml`;
the README's `liga.yaml` command does not match the released file.

## Li et al. — Stereo R-CNN, CVPR 2019

[Official code](https://github.com/HKUST-Aerial-Robotics/Stereo-RCNN).
Authors: Peiliang Li, Xiaozhi Chen, Shaojie Shen. Required original-paper
downstream detector. The repository offers legacy PyTorch environments and
pretrained-weight links. Must isolate its environment from the new detector.

## Liu et al. — RDcomm, ICLR 2026

[OpenReview](https://openreview.net/forum?id=920RxFvsMx),
[official implementation](https://github.com/gjliu9/RDcomm).
Authors: Genjia Liu, Anning Hu, Yue Hu, Wenjun Zhang, Siheng Chen.
[Proceedings full text](https://proceedings.iclr.cc/paper_files/paper/2026/file/c32aaa92fcd18a0482d2800473cf1894-Paper-Conference.pdf)
read in Sections 3–4 and Appendix A.6. Defines pragmatic distortion through
increase in conditional Bayes risk, with receiver observations and multi-agent
redundancy. Implements layered codebooks, task-prioritized variable length
coding and mutual information selection. Detection derivation assumes independent
locations, simplified focal loss and conditional Gaussian regressands. These
assumptions must not be silently transferred into a stereo JSCC theorem.
Our scalar Gaussian power proxy is not an extension/proof of their minimum
bit-rate formula. Direct task decoding, uncertainty and task rate-distortion
are already represented here; stereo depth structure and physical channel
allocation are candidate narrower differences. Code/accounting audit still open.

## Gan et al. — CoDS, TWC 2026

[IEEE record](https://ieeexplore.ieee.org/document/11554267/),
[full preprint](https://arxiv.org/html/2512.22513), 27 December 2025 v1;
IEEE publication 8 June 2026, DOI `10.1109/TWC.2026.3698559`.
Authors: Jipeng Gan, Le Liang, Hua Zhang, Chongtao Guo, Shi Jin.
Sections II–IV describe LiDAR/PointPillars collaborative features, learned
quantization, LDPC/modulation and task training. Receiver reliability uses
decoder LLRs and semantic damage to filter failed digital features. This differs
from a transmitter stereo depth posterior, but defeats any generic claim that
task training or uncertainty makes our system new. Their comparisons constrain
complex channel uses; analog symbols must not be called digital bitstreams.

## Zhao et al. — Real-time mobile 3D reconstruction, arXiv 2026

[Author full text](https://arxiv.org/html/2607.16128), 17 July 2026 v1.
Read Sections III–VI. Monocular image communication jointly decodes RGB and
confidence; confidence targets derive from pixel reconstruction error. Losses
combine MSE, descriptor consistency and confidence supervision. Receiver
confidence weights RANSAC sampling and bundle-adjustment residuals. Evaluation
uses rendered multi-view scenes and pose/reconstruction metrics, rather than
KITTI stereo detection. This overlaps broad geometry/reliability claims.
Transmitter depth posterior allocation and direct detection latents remain
candidate narrower distinctions. Their BLUE argument is conditional on the
assumed residual covariance, not proof of calibrated learned confidence.
Do not equate its reconstruction/pose task with stereo object detection or
declare non-overlap from task names alone.

## Men et al. — Channel-adaptive stereoscopic media, TCCN 2026

[University of Surrey record](https://openresearch.surrey.ac.uk/esploro/outputs/journalArticle/Channel-adaptive-Semantic-Communication-for-Stereoscopic-Media/991126376502346),
DOI `10.1109/TCCN.2026.3689868`. Authors: Jingxuan Men, Mahdi Boloursaz
Mashhadi, Ning Wang, Yi Ma, Mike Nilsson, Rahim Tafazolli. Verified abstract
describes dual-view JSCC, correlation/importance/channel-dependent attention
masks for adaptive rate, reconstruction metrics and a 3D rendering prototype.
Full text has not been obtained; overlap assessment is abstract-level only.
Consequently, stereo correlation, channel adaptation and feature importance
cannot alone support novelty. Direct detection supervision and depth-risk
allocation are candidate differences requiring full-text and empirical audit.

## Original-paper implementation recovery

The appendix specifies staged training of semantic, flow, fusion and channel
networks. The global transmitter table lists inverse pixel shuffle producing
144 channels, followed by a convolution accepting 256: unresolved inconsistency.
No silent correction can establish exact reproduction. Detector version, exact
split, ROI side information and channel accounting also remain open.

The official Stereo-RCNN Python 3 branch is pinned at
`4d6dd65049f52c1a5f2b6ad716a7e0da5cb02cb3`; master uses different legacy ROI
sampling. Weight shape agreement alone will not establish equivalence to the
original communication paper's detector. Runtime migration preserves the
branch's full CUDA ROIAlign, inclusive-pixel NMS and dense alignment coordinates.

## KITTI and conference rules

[KITTI 3D benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).
Use official-compatible AP_R40 with Car IoU 0.7 as primary new-study metric.
Keep original-paper IoU 0.5 replication as a separate protocol.

[Official IEEE ComSoc ICC 2027 page](https://www.comsoc.org/conferences-events/ieee-international-conference-communications-2027):
30 May–3 June 2027, Washington DC; technical-paper deadline 2 October 2026.
Conference site was inaccessible to the browsing tool. Extension, EDAS closing
time, page limit and workshop windows are unverified. Do not infer availability
from third-party dates or the unrelated `icc27.org` site.

## Candidate references supplied by user

WhisperNet's primary abstract and method source were checked on October4;
see [the scoped reading note](WhisperNet-2026.md). Receiver-coordinated spatial
and channel requests are existing work and remain an overlap constraint.

Where2comm, How2comm, MRCNet, HEAL, CoSDH, GSCOOP, CoopTrack,
PragComm, WhisperNet, CoDS, Mono2Stereo, TasCom, CATNet, CoopDiff and SToRe3D
are queued for source and overlap checks. Their supplied descriptions and
claims of non-overlap are not yet established research evidence.

## 2026-10-03 follow-up source check

IEEE ComSoc official ICC 2027 page still lists October 2, 2026 technical-paper
deadline; no extension verified. Target remains ICC 2027 pending user preference
and actual submission window, not silently switched.
https://www.comsoc.org/conferences-events/ieee-international-conference-communications-2027

No original communication implementation found in the repeated title/code
search. This does not prove code absence. Primary original article remains
https://arxiv.org/html/2502.12735v1 . Exact codec reproduction stays open.

## 2026-10-03 - Additional primary full-text access

Bristol's author repository provides an accepted-manuscript PDF for Men et al.
TCCN2026, previously unresolved at Surrey. Metadata gives published4May2026,
volume12, pages7909-7925 (17pages), DOI10.1109/TCCN.2026.3689868.
Full method/accounting assessment is still pending until the manuscript is read.
https://research-information.bris.ac.uk/en/publications/channel-adaptive-semantic-communication-for-stereoscopic-media-de/

The linked accepted-manuscript PDF returned403 to both browser-fetch and
ordinary curl on2026-10-03; the university metadata is verified but the full
manuscript has not been obtained/read. Preserve the abstract-level limitation.
Direct author repository link: https://research-information.bris.ac.uk/files/486881541/IEEE_TCCN_Final.pdf

CodeFilling and newly located Video TokenCom primary methods have now been read;
their specific overlap and remaining comparison/accounting gaps are recorded in
[novelty-gates.md](novelty-gates.md). No adapted implementation or reproduction
claim follows from this literature reading.

## 2026-10-04: receiver-alignment and geometry-coordinate check

New primary GraftNet(CVPR2022) method content reviewed; limited relevance and
lack of wireless/detection reproduction recorded in [graftnet-2022.md](graftnet-2022.md).
Direct method text supports prior cosine-cost/feature-adaptor/aggregation
retraining work. Do not claim those generic mechanisms as our novelty.
A fresh stereo/semantic/3D title search primarily returned the original Cao
paper and known Men TCCN2026 work; this is not evidence of absence or novelty.
The original paper and Men abstract claims remain distinguished from our
independent implementation and unknown author-code identities.

Our native depth-cost axis is metric-depth sampled, not uniform disparity;
[geometry-coordinate-system.md](../docs/geometry-coordinate-system.md) derives
only its sampling/pooling arithmetic from current frozen code. No rate-distortion
or AP theorem is claimed. Structured depth coding, receiver adaptation and
preventable-damage calibration remain untested hypotheses pending full diagnosis.

## 2026-10-04: ECSIC actual-bitstream audit

Official ECSIC source revision696f4ae4f250bb1fc750ae9ec23e9c98c2c7e6db
was pinned and inspected. Its evaluation uses estimated negative-log-likelihood
rates; this release exposes no arithmetic/rANS serialized encoder/decoder API
in the inspected15 Python files. Estimated bpp cannot become LDPC input bits.
Actual coder, finite CDF/decode order, weights and KITTI adaptation remain to be
resolved before an ECSIC digital-channel result. This is a release-specific
implementation gap, not a claim that the underlying method cannot be coded.
See [ecsic-2024.md](ecsic-2024.md) and the retained source audit.

## 2026-10-04: actual LDPC comparator implementation

Official pinned NVIDIA Sionna PHY source and documentation read; independent
CPU-only runtime and actual NR LDPC/QAM/APP/BP transceiver implemented and run.
The native source-stream identity checks pass and noisy synthetic failures are
retained. NR family/n1944/BP20 are declared choices because original paper
settings remain unpublished. Detailed source/relevance note:
[sionna-digital-2026.md](sionna-digital-2026.md).

## 2026-10-04: importance and power-allocation overlap

Primary sources add [ISFR](isfr-2026.md), [Generative Feature Imputing](generative-feature-imputing-2025.md)
and [SIAC](siac-2023.md). Importance ordering, task/semantic scoring and unequal
power protection already have precedents. A future stereo resource-exchange
policy needs a specific mechanism and matched generic controls; none of these
search results proves novelty or absence of competing stereo methods.

## 2026-10-05: PragComm method-version and publication check

Publisher confirms TPAMI48(8),9279–9296 and9April2026 publication,
DOI10.1109/TPAMI.2026.3680062. Reviewed full primary text is explicitly2024v1,
not asserted identical to the final edition. Source scope, direct-feature/
dictionary overlaps and the version's noiseless abstraction are recorded in
[PragComm-2026.md](PragComm-2026.md). This prevents generic direct task coding
or adding a physical channel from being promoted alone as our contribution.
RDcomm official proceedings and Men author-repository metadata were rechecked;
this does not change their existing evidence or Men full-text limitation.
