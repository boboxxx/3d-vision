# F2 native task adaptation from the audited F1 student (lock before launch)

F1 completed five feature-only epochs. On the fixed372-frame training-only
holdout, independent native uncompressed Car moderate3D AP_R40=11.0028721%,
versus F0 uncompressed0% and clean-author99.8629021%. Appearance NMSE remains
.665291. Partial feature agreement does not establish useful final wireless AP.

Question: can direct native task gradients adapt the existing student to the
unchanged complete receiver? Do not change receiver capacity, parameters,
feature interfaces, dataset or augmentations. This is exploratory, not an
allocation comparison. Preserve negative outcomes and the earlier F1 result.

Initialize every original and student tensor from F1 epoch5 checkpoint SHA
cefa61a9b2f18376109b156044442d14706f4f631e1e88810bca016a0f283334.
Use weights only: a fresh AdamW optimizer, no F1 optimizer moments or epoch
continuation. Fresh seed17, fixed3340/372 train-only fold, batch1, four workers,
one complete3340-returned-batch epoch. Native empty-augmented-GT replacement
sampling is retained; do not claim every returned frame is unique.

Train only the same35 student parameter tensors. Use the full original native
training forward/loss: 3D classification/regression/direction/IoU, depth, 2D and
LiDAR imitation, plus existing student normalized feature loss at weight0.1.
This includes auxiliary teacher supervision; it is not detection-loss-only.
No semantic link, noise, posterior or sensitivity loss. Receiver/teacher
parameters and BN statistics remain frozen. Native global_step increments3340;
only the two native training-only imitation-scale buffers may additionally
change. All other484 original-state entries must exactly match F1 initialization.

AdamW LR0.0001 constant, weight_decay0.0001, gradient clip10; reject nonfinite
loss/gradient before update. Save full author-compatible checkpoint and native
TensorBoard scalar events. Independently audit CRC, all3340 loss/preclip/feature
records, optimizer states, source/split/initialization identities and allowed
original buffers. All comparisons must charge the five F1 feature epochs and
this task epoch equally; no main-validation selection is allowed.

The native trainer produces an automatic holdout evaluation; preserve it.
After training/state audit, separately run strict sensor-only uncompressed
native evaluation at seed17 on all372 IDs, then independently audit prediction
files, raw labels/calibration and recomputed AP. Both evaluations should agree.
Final checkpoint selection is fixed at the only completed epoch. Do not extend
training based on these metrics without a new explicitly logged protocol.

Interpretation: improvement tests adaptation of this fixed student, not the
communication hypothesis. If AP remains inadequate, separate sender-capacity
or receiver-adaptation investigations may follow. Power/codec comparisons
remain deferred until their matched representations are established.
