# Independent parameter count versus state dictionary aliases

The prepared formal auditor002 incorrectly assumed225 independent trainable
parameters because the official model has225 state_dict entries. It has not
been executed; preserve it and its prelock. The complete engineering GPU report
and native/local array audits001 instead confirm216 named parameters totaling
31640742 values, and225 registered state entries. Nine right E0–E5 parameter
entries alias the shared left convolution/PReLU entries, as official Stereo
with shared=true constructs. This is expected public architecture behavior.

Isolated auditor003 corrects only named-parameter/gradient/optimizer tensor
counts225→216. Complete225 model states, all37120 update records, all10 epoch
checkpoints, exact hashes, finite conditions and numerical loss bounds remain.
Additionally require all nine right/left shared entries to be exactly equal
in public initialization and every checkpoint. No training/model/source,
lambda/seed/exposure/selection criterion or numeric tolerance is changed.

Evidence: ecsic-KITTI-GPU-report-001.json, both complete native/local144085094
saved-value audits,126562968 gradients and formal candidate0's live report.
The initial formal training code correctly iterates model.parameters and
records216 tensors; there is no reason to restart it. Future formal closures
use003 and explicitly retain the scope limit: whole logs/epoch states, not
independent full raw-gradient or per-update Adam replay.
