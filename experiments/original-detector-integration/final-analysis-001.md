# Full trained original RGB variant: six internal endpoints

Complete83epochs/277220updates, allfive full-state/Adam/raw training audits and
sole final768 checkpoint08a7c848…49daf674 preceded execution. Six prelocked
endpoints and both shared372 received caches are now fully server-audited and
terminal; wrapper56638/cycle56656 actually exited. Local verifier362a05f checks
all32 retained artifacts, full2982 prediction/GT/calibration/metrics files,
source identities, all768/670/484 readonly-state metadata and paired actual
received tensors. Server independently rereads predictions/GT and recomputes AP;
local verifier does not rerun AP or transfer28GiB received float tensors.
Evidence/result commit7bc6c27. Unique failures and former snapshots stay retained.

| Fixed detector | Clean relay Moderate3D | Recovered identity | Recovered AWGN10 |
|---|---:|---:|---:|
| Stereo-RCNN | 40.49105380 | 40.53297440 | 41.76036257 |
| LIGA-Stereo | 99.86290210 | 79.36053794 | 75.92211593 |

CarIoU.7 AP_R40percent, fixedinternal372. CompleteEasy/Moderate/Hard,2D/BEV/3D,
secondaryIoU.5 R11/R40 are retained in every endpoint. Shared decodedfloatRGB
inputs are exact acrossbothdetectors; cleanreferences use exactnativeimages.
No clipping, retry, cleancopyrefinement or GT/depth/traininghead atreceiver.
Actualzeroerasures in both caches; successfuldelivery does not imply unchanged
image/taskinformation. Sensor-dependent complexduration totals74424347/372 =
200065.44892473117mean, actualenergy74424346.53516832 for eachcondition; controls
and fullfixedROI resources counted. All372 attempts retained in AP denominator.

For LIGA this operatingpoint loses20.5024pp under identity reconstruction and
23.9408pp underAWGN10 relative to its cleanreference. StereoRCNN's modesthigher
noisyAP is an observedsingle-seed result, not a beneficial-noise theorem or
statisticalgain. Detectors have different response to the SAME receivedimages;
that does not mean channelinformation differs byreceiver. Internalpretraining
includes372frames, explaining why LIGAcleanreference is notmainvalgeneralization.

Declaredsource/staging/geometry/ROI/control/settings choices differ from unknown
authorimplementation; this is not exactpaperreproduction. Newdirect3D F7 uses
62400symbols rather than200065mean and differenttraininghistory. Comparing75.92
with24.53 cannot establish either method's rate/energy/exposure superiority.
The next main comparison must matchdetector, operatingpoint, realizedenergy,
trainingexposure and originalIoU.5 evaluation before claimselection.
