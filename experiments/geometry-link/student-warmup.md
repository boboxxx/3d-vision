# F1 feature-only sender pretraining (lock before launch)

Motivation: D0 independently recomputes clean/learned-student uncompressed
Car moderate3D AP_R40 IoU0.7=99.8629021/0.0% on the same372 codec holdout.
F0 student distillation loss stayed near its zero-output reference even though
joint native training loss decreased. Useful sender features are not established.
This diagnoses a failure before communication; it does not prove the codec is
sound or identify a unique cause. Do not run a power-allocation study yet.

F1 is exploratory. Initialize the full original detector/feature teacher from
the same released author checkpoint, with a fresh student seed17 (not F0's
student or optimizer). Preserve original feature interfaces and all original
receiver parameters. Train only student parameters, directly matching teacher
left/right stereo and left appearance features through the existing equally
weighted normalized MSE. Teacher always eval/no_grad; all original parameters
and buffers remain fixed. No channel codec or full 3D loss in this warm-up.
It is feature pretraining, not the final communication training objective.

Use the fixed3340/372 original-training-only fold, native paired image crop/
augmentation/normalization, batch1, four workers, five complete3340-step epochs,
AdamW LR0.001 constant, weight decay0.0001, gradient clipping10, reject nonfinite
loss/gradient before update. Save each epoch checkpoint and immutable scalar
records, total16700 updates. No main validation accesses for selection.
Record actual parameter names, interface RMS, per-interface normalized MSE and
cosine agreement with teacher; identify scale or collapse without assuming it.

Implementation clarification before F1 launch: retain native LIGA replacement
sampling when augmentation leaves an example with no GT boxes. Each complete
epoch is3340 returned batches/updates, not a claim of3340 unique returned IDs.
Record actual returned IDs and unique counts; every returned ID must remain in
the locked training fold. Feature functions receive only stereo RGB; native
loader still reads labels/LiDAR to preserve its sampling and augmentation.
Holdout feature metrics require all372 IDs in exact native order. Preserve RNG
states around each holdout pass so it does not perturb later training epochs.

Evaluate feature agreement on372 holdout frames after each epoch and complete
uncompressed native3D AP after epoch5. Keep the fixed five-epoch budget even if
early metrics improve; any continuation is a new exploratory protocol. Preserve
negative outcomes. Check all original state tensors for exact equality after
the known sparse layout conversion; original globalstep does not increment
because native detector training forward is not used. Strictly load all
original/student weights for inference. No inference teacher or GT inputs.

Only if useful sender performance is established, use one shared frozen student
initialization for identical uniform/geometry/task controls, then lock codec
reconstruction/task warm-up and joint receiver budgets. Count these pretraining
exposures/teacher compute in every matched scientific comparison. A failed
feature match may require a separately scoped sender-capacity investigation;
do not silently replace the full receiver or claim communication gains.
