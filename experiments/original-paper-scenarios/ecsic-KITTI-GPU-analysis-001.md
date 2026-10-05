# Native ECSIC source-RD execution and KITTI baseline progress

The official ECSIC model now runs on a separately allocated RTX PRO6000.
GPU11425032, complete native audit11425041 and complete-stream export11425042
all actuallyCOMPLETED0:0. Local stream process also exited0. Two full-native
KITTI training pairs000000/000003 produced144085094 saved values and126562968
raw/clipped gradient values, all checked natively/locally. Independent FP64
RGB distortion, estimated-rate RD loss and complete Adam transitions pass the
prelocked limits. Peak allocated5.42GiB/reserved5.87GiB, not an edge-cost claim.

The public model has225 state entries and216 distinct parameter tensors with
31640742 values; nine right encoder entries alias shared left parameters.
The planned formal auditor002 confused these counts, before its execution.
Preserved002 and isolated003 correct the parameter counts and additionally
check all nine alias values, without changing training or any numeric limits.

Official neural sources and public checkpoint remain unchanged. The separate
pure-Python einops copy and read-only torchvision overlay preserve the active
runtime; an import-only disabled wandb module prevents experiment-management
or external logger calls. This bridge is disclosed, not the author's training
CLI or unpublished Cao experiment environment.

Formal KITTI adaptation is a single-seed17 six-lambda rate sweep with10 full
3712-pair epochs/candidate and final-epoch-only selection. Array11425047 runs
one candidate at a time; first rawjob11425048 actually reached5824 updates,
RTX PRO6000/no traceback. These are operating points, not multiple seeds.
Only complete final weights and normal terminals qualify for fixed64 training
pair actual rANS calibration. Estimated bpp here cannot stand in for bytes,
nominal10/30/50 compression or actual LDPC/QAM channel uses.

The source baseline receives RGB reconstruction supervision; the proposed
cost-field link receives native3D task supervision. Compare pretraining,
adaptation exposure, sender compute and downstream detector explicitly.
No new-method AP or communication advantage follows from this engineering
gate. Full source/radio caches, exact actual-resource comparisons and both
detectors remain required. Formal logs/epoch states also do not claim every
raw CUDA gradient or every Adam transition has independent replay evidence.

Evidence: ecsic-KITTI-GPU-terminal-001.json, native/local-audit-001.json,
GPU-report-001.json, formal-launch-002.json, and
cost-field-ECSIC-single-seed-actual-snapshot-008-001.json.
Reference: [official ECSIC implementation](https://github.com/mwoedlinger/ecsic).
