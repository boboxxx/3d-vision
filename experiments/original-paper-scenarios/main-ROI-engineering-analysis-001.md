# Complete sensor-only ROI geometry engineering: closed

Protocolc73eada fixes public ROI and explicit future PSNR/SSIM conventions
before new ROI/quality inspection. Four isolated files9b156bb reuse pinned
YOLOv5n v7.0, checkpoint4f180cf2, official CPU FP32 letterbox/NMS/scale_boxes,
car/bus/truck0.25-confidence and0.45-IoU thresholds. Exact Cao weights and
thresholds remain unknown; this is a declared ROI variant, not author ROI
reproduction. Old train/internal-fold extractor and all active roots stay fixed.

Four CPU fixture families pass on local and sheng: literal pixel oracle,
independent prefix-sum union, invalid geometry and actual forbidden sensor/
GT/cache reads. Engineering269478 then finishes CPU-only extraction and
fresh independent audit on000000/000003, both cameras. Each source PNG is
decoded through CV2 from audited bytes and agrees with independent PIL RGB;
the native auditor verifies all5,511,780 actual RGB values and independently
rebuilds every saved rectangle union. All121 parameter/buffer state identities
remain unchanged; no optimizer, GPU, GT, calibration or detector is used.

Four views contain4boxes total and one empty view. Empty key quality remains
undefined; do not drop the view or assign perfect/zero quality. Actual process
exit, both successful commands and eight-artifact closure are verified. Local
verification hashes all eight transferred artifacts and reconstructs all four
union masks from saved rectangles, without claiming local raw PNG/YOLO replay.

The once-only full public3769pair/7538view CPU cycle269672 is now running with
the engineering proof uploaded. It shares no GPU with main detector269262.
Full independent PNG/mask audit, native terminal and local all7538-record
verification remain before quality use. PSNR/SSIM operations are prelocked
but not yet implemented/measured, and ROI does not enter JPEG/JP2 coding or
current detector inference. Original full wireless matrix and new-method
geometry-specific evidence remain unfinished.

Evidence: `public-val-ROI-{local,sheng}-CPU-001.json`, engineering001 native
manifest/records/audit/closure and`public-val-ROI-engineering-001-local-verification.json`.
