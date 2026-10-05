# Final trained original RGB baseline

Protocol: [final execution details](final-code/evaluation-protocol.md), supplementing
[parent integration protocol](protocol.md). Implementation commit `0f6d65c` follows
protocol commit `7e6f3e4`. Local and sheng CPU contract checks (5 each) passed;
this does not establish native GPU execution or trained AP.

The first queue001 completed the original StereoRCNN GPU probe, then failed
before F7's first optimizer update at a strict CUDA device check. Its follow-on
receiver queue stopped without running either new receiver probe. Both failures
are retained and terminal; neither is treated as a passed prerequisite.

After separately locked device repair,19 CPU checks passed on local and sheng.
Recovery native-validation-queue-002 (PID54337) waits for the actual original
cycle to terminate and the complete stage5 audit to pass, then runs fresh F7
sanity002, audits and terminal closure, followed by both new receiver GPU probes
on000036. Child probes prove actual parent-queue/process identity; formal AP
requires the entire queue to be terminal and passed. It stops after engineering.
No formal F7 or original AP is launched by this queue.

All root/upstream executable identities, existing adapter/probe sources, F7 code
and final-code sources are frozen until their queued engineering closes. The
original21 sources/protocol remain frozen for the entire original cycle. Reports
and this README lie outside executable source identity scopes.

After both native receiver probes pass, seal/sync/review and commit their evidence.
The full original5-stage cycle must have terminated, all83 epochs and independent
audits must pass, and the sole full768 final stage5 epoch45 checkpoint must match
the complete predecessor chain. `common.final_original_chain()` rejects live
training/partial weights/missing native engineering or changed source identities.

Only then execute, with a unique prefix and the existing runtime:

```bash
cd /home/sheng/paper6
source scripts/sheng_env.sh
python experiments/original-detector-integration/final-code/cycle.py --prefix original-final-native-seed17-001
python experiments/original-detector-integration/final-code/close.py --prefix original-final-native-seed17-001
```

The cycle first encodes/decodes each fixed372 frame exactly once per air condition
(identity, AWGN10), using a dedicated CPU Generator17 in frame order. Lossless
receivedfloat32 pairs are retained on/mnt/d, with source/tensor/file/noise-state
hashes. An independent cache audit checks every actual array, raw sensor identity,
ROI-union/header resource formula, erasure and noise RNG replay. Cache space is
about28GiB total, plus retained outputs; large tensors never go on the root disk.

Both author detectors read the same audited received tensors, with no codec or
clean images at refinement/alignment. Six fixed endpoints comprise each detector
under clean relay/identity/AWGN10. LIGA's native Pedestrian/Cyclist outputs remain
in the text files; the prelocked primary metric is Car3D IoU.7 AP_R40. Erasures
produce empty prediction files, zero detector calls and retain all attempted
physical resources and the full372 AP denominator. No retries, clipping/uint8,
new training, threshold selection or best checkpoint occurs.

Each endpoint has a fresh all-file prediction/GT/calibration/read-only/native-call
audit and independent AP recomputation, including secondary IoU.5 R11/R40. The
cycle verifies paired received identities across the two detectors. `close.py`
requires the actual cycle PID to be terminal and verifies source, cache/prediction/
GT/calibration/metric/audit/log hashes before releasing the final execution freeze.

These are exploratory internal372 results with detector-author pretraining overlap,
not mainval. The declared9-channel reconstruction variant is not exact author-code
reproduction. Its sensor-dependent roughly200k mean complex-use operating point
and direct3D62400 operating point have different exposures/rates/energy/compute;
these six endpoints alone cannot support a fair superiority/novelty claim or
complete the project's risk/allocation/fading/seed/mainval/manuscript gates.

## Latest verified execution

Full original83epochs and all5stage audits close with sole768 finalweight08a7c848…49daf674. Repairednativeengineering and formalF7 both close. Originalfinal sixendpoint cycle is now CLOSED: all6×372nativeevaluations, both372sharedcache audits, fullreadonly/pairedreceived/AP/source/file closure, actualPIDs56638/56656terminal. Localcompleteverifier362a05f passes32retainedartifacts and2982prediction/GT/calibration/metrics files. LargefloatRGBtensors stay onsheng; localverifier doesnotrerunAP.

ModerateCar3D IoU.7/R40: StereoRCNN clean/identity/AWGN10 40.4911/40.5330/41.7604; LIGA99.8629/79.3605/75.9221. Mean200065.4489complexuses/frame and actualrealizedenergy counted; zeroactualerasures. Single-seed/internalpretrainingoverlap/rateandtrainingmismatch stillapply. No fairnewmethodsuperiority claim. [Fullresultanalysis](final-analysis-001.md) and data/provenance/original-final-native-seed17-001-local-verification-001.json giveevidence. Do notlaunchduplicates. Sealedexperimental sources must remain unchanged for provenance and any newpilot reuse.
