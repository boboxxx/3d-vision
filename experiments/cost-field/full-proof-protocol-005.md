# Full training record audit, separate from active training inputs

Audit code005 is additive. Do not alter training004, its source lock, weights,
schedule, records or running jobs. All checkpoints load with weights_only=True.

The complete gate requires 44544 updates per seed, all twelve completed epochs,
the fixed final-epoch checkpoints, all original484 states and actual Slurm raw
job COMPLETED0:0. An explicitly counted immutable live prefix may test the
audit machinery; it is never full training closure or validation evidence.

Stream every saved array, raw source image/calibration/label and per-step
optimizer checkpoint; retain all native artifacts. Local processing holds one
record, source metadata and small codec states, without persisting large raw
arrays. Ordered tar is data transport only; never extract or execute its paths.
Check exact SHA, dtype, shape and all finite values, no sampling. Preserve
failed proofs and their checked-record ledgers.

Independently regenerate epoch permutations and channel/SNR schedules from
the locked seed. Check source hashes, camera labels/aliases, pseudo-lidar
coordinates, crop metadata, positive-anchor counts, scalar loss arithmetic,
model/cut hooks, before/after state chains, gradient clipping and every Adam
transition. Anchor target descriptors must agree for every repeated frame;
this checks deterministic targets, not independent anchor assignment.

Adam reference starts each transition from the actual previous FP32 parameter
and moments, calculates the transition in FP64 and uses a fixed128*FP32eps
operation envelope. This is not a counterfactual long FP64 training trajectory.
Moment bounds use sum of contribution magnitudes; parameter bounds include
old parameter, update and result magnitudes. Clipping norm bound64eps*(1+norm),
clipped gradient bound8eps*abs(reference)+eps. Empty/zero-gradient frames stay.

Check every AWGN/Rayleigh complex multiply and untruncated perfect-CSI ZF
element in FP64, propagating64eps sum-of-magnitude FP32 arithmetic envelopes.
Validate all19-symbol site energy groups with1e-6 relative tolerance, and
reported total energy at1e-8 absolute tolerance. No fitting numerical limits
to observed errors. Record noise/fading moments; no empirical distribution
threshold selects or discards training records.

Raw camera labels compare exactly after FP32 parsing. Converted GT uses
64eps*(1+abs(reference)) envelope. Range corners use64eps*(1+abs(center)+
sum(dimensions)+max(abs(range))) around each coordinate: robust decisions
must match, ambiguous decisions remain explicitly counted. Cropped projection
matrix check64eps*(1+abs(reference)+abs(original)) includes FP32 inverse/matmul.

All saved arrays and physical/optimizer transitions are checked. Original
processed RGB pixels and full decoded cost fields were not saved; this audit
does not independently replay them or certify gradient correctness. Full
validation/AP and original radio/rate controls remain separate gates.
