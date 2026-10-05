# Novelty and evidence gates — working audit, 2026-10-03

The candidate contribution is stereo depth-risk allocation for direct detection
under a measured physical-channel budget. It is not established by the current
implementation or negative F0 result. A change of dataset/task name alone does
not establish novelty.

| Existing primary evidence | Already covered | Candidate narrower question | Evidence still required |
|---|---|---|---|
| [Cao, TCOM2025](https://arxiv.org/html/2502.12735v1), Sections II–IV | Stereo correlation, semantic ROI, channel codec, detection evaluation | Can a lightweight sender communicate usable detection features directly? | Documented original Flow/ROI reproduction; same detector or controlled receiver comparison |
| [RDcomm, ICLR2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/c32aaa92fcd18a0482d2800473cf1894-Paper-Conference.pdf) | Task distortion/Bayes risk and prioritized codebooks | Does stereo depth structure improve allocation under noisy physical channels? | Assumptions, calibrated risk, matched generic task-allocation baseline |
| [CoDS, TWC2026](https://arxiv.org/html/2512.22513) | Detection features, task learning and digital reliability | Does sender stereo uncertainty predict preventable 3D damage? | Distinguish intrinsic ambiguity from channel damage; physical-resource controls |
| [Men, TCCN2026](https://research-information.bris.ac.uk/en/publications/channel-adaptive-semantic-communication-for-stereoscopic-media-de/) | Abstract confirms inter-view/importance/channel masks and adaptive rate | Detection-derived depth risk rather than reconstruction importance | Full text still inaccessible; do not assert complete non-overlap |
| [CodeFilling, CVPR2024](https://arxiv.org/html/2405.04966v1), Sections 4.2–4.3 | Detection-driven compact codes and spatial selection | Stereo geometry before BEV plus noisy-channel allocation | Adapted control with the same features/compute; sparse-index and score-map overhead |
| [Video TokenCom, arXiv2026](https://arxiv.org/html/2603.02470v1), Sections II-B–II-E | Importance-dependent precision, unequal protection and source/channel selection | Detection-derived priority without textual intent or video reconstruction | Do not claim generic variable rate or unequal protection is new |

CodeFilling full primary method was read via the author preprint after the CVF
PDF fetch failed. Its confidence maps support information-demand selection;
task-trained codebook indices reconstruct BEV features for detection. Those
score disclosures are part of the method, so an adapted comparison must account
for them. Its multi-agent BEV setting differs from our stereo sender/cloud
setting; this is a scope distinction, not proof of algorithmic novelty. No
physical channel reproduction or author-code accounting audit has been done.

Video TokenCom was newly located through the author's homepage and read in
Sections II-B–II-E and III. CLIP/flow determine intended token classes; discrete
video token precision and MCS are selected per class under resource/reliability
constraints. The receiver reconstructs video. The source explicitly discusses
PDU overhead and mask/motion side information. This overlaps proposed variable
bits and protection, even though our implemented link currently changes power
only. Its preprint status is not a verified conference/journal acceptance.

Current F1 is an exploratory feature-initialization repair. It tests neither
geometry allocation nor any novelty claim. All subsequent matched controls must
share its initialization and charge the same teacher exposures/pretraining.

## Current method scope,2026-10-05

Historical F0/F1 notes above are not the current full-training admission.
Candidate004 is a learned conditional metric-depth field at a fixed physical
budget; onlyseed17 is current. G/P/S/B all use the same frozen task receiver,
with generic dense B as an architecture control and G/S sharing P's field
architecture. P-B cannot alone isolate the native stereo prior; use P-G/P-S
for that question and retain the negative internal F9 primary result.
[PragComm's specifically reviewed2024v1](PragComm-2026.md) already covers
direct task features, dictionary coding and selective sharing. A phase-noise
calculation or different stereo setting alone is not an empirical contribution.
Exact packet-coordinate moments are mechanism analysis, not calibrated task
uncertainty, geometry sufficiency, detection error bounds or new RD theory.
All84 full validation endpoints, original RGB/flow/radio controls and actual
resource/compute accounting are still needed before final scientific claims.
