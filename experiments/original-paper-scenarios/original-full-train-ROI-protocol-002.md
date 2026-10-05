# Full public-training ROI admission for the original RGB/flow variant

Locked before execution. The existing independent YOLOv5n v7.0 sensor-only
caches cover disjoint internal subsets3340 and372, together exactly the public
KITTI train3712. Reuse all saved detections unchanged, concatenate by the public
training-ID order, and retire the internal holdout role for this new full-split
baseline. Never include any public validation frame or use its AP to choose ROIs.
This is a declared YOLO checkpoint/threshold variant, not the original authors'
unavailable detection cache or communication weights.

Require both original extraction manifests and independent audits, exact source
SHA identities, original fold identity, the complete raw KITTI manifest004, and
YOLO metadata. The merger preserves every view field and box in every row;
verify complete unique IDs, source PNG identities, sizes, box class/confidence/
coordinates, mask unions, and mask SHA256 for all7424 views. Native audit reads
all7424 original training RGB PNGs, verifies bytes/SHA and RGB dimensions against
the prior whole-data manifest, and independently recomputes union masks. Read
no labels, calibration, LiDAR, received images or public validation PNGs. Metadata
includes hashes of other files; only selected sensor PNGs are opened by this task.

Use CPU two threads on sheng; never interrupt the live SRCNN process. Outputs
are exclusive under data/engineering/original-full-train-ROI-002. Repeat the
complete metadata/box/mask audit locally on transferred original sources, merged
records and native audit. Local audit does not reopen unavailable raw PNGs;
source byte proof is the already verified whole-data manifest plus native reads.
Record those distinct scopes and require identical ordered record ledger SHA.

This closes input admission only. It does not train a new model, reproduce AP,
measure actual radio resources, prove fresh YOLO inference, or change frozen
reproduction/cao2025 sources. The original36/4 full-training variant and36/1,
64/16 architecture variants require their own prelocked runs and actual terminal
closure. The old83-epoch3340 internal run cannot be relabeled full train3712.
