# Independent native test-mode class mapping repair002

2026-10-04, before fitting. Complete audit attempt002 passed the repaired
unchanged-tolerance gradient reference but stopped on training-frame000012:
one recorded native target versus two expected targets. Preserve the failure
log and verifier identity; no input, model or observation changes.

Source inspection resolves the discrepancy: StereoKittiDataset.__init__ sets
use_van=USE_VAN and training, and use_person_sitting=USE_PERSON_SITTING and
training. This diagnostic explicitly uses training=False. Therefore Van and
Person_sitting keep their names and are excluded by the class-name selection,
whereas the two-frame independent helper erroneously assumed their training
mapping to Car/Pedestrian. The omitted second000012 label is Van. Point-cloud
box-range filtering is training-only and does not explain this discrepancy.

Add a dedicated independent full64 input reference with the exact native TEST
classes Car=1, Pedestrian=2, Cyclist=3. Keep all coordinate/yaw/calibration/RGB
checks and tolerances unchanged. Leave the previously sealed engineering helper
unaltered, so its audit and eligibility identities remain valid. No GT selected
by losses or outcomes. Audit every one of the same64 scenes and all8320 rows.
Continue to retain the two prior failures and forbid fitting before full closure.
