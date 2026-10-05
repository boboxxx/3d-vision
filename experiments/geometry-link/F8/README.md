# Matched student encoder adaptation, F8

Formal `stereo-encoder-native-seed17-001` is fully closed; cyclePID167748
actually exited. Both3340-update arms and four372-frame endpoints pass all
independent audits and local46-artifact/6680-row verification. AWGN10 Moderate
AP22.3785% codec versus25.6649% joint; prelocked difference+3.2864points.
See [analysis](analysis.md) for complete scope and limitations.

The [protocol](../F8-matched-encoder-adaptation.md), commitad64d76, fixes the
fullF7AWGN parent ea93fb1e…4de36a3,3340updates per arm, fresh AdamW1e-4,
SNR1717/noise1718 and62400complex uses. All535 states load exactly. Control
trains16codec tensors with519 fixed; joint trains51 with484 original detector
states fixed. Same inference architecture; joint training uses more backward
computation. Extra optimization is a baseline control, not a novel contribution.

Seven CPU checks pass locally and onsheng with identical nine-file source
identities. A prelaunch root-path error was preserved and narrowly repaired;
no native updates occurred before that correction. See the separate repair
protocol and `data/provenance/F8-prelaunch-path-failure-001.json`.

Engineering `stereo-encoder-native-sanity-001` closed12actualupdates, complete
state/Adam/noise audits, exact paired augmented inputs and local verification
of all14 retained artifacts. Eacharm includes one legalemptyGT case. All selected
parameters receive finite learning signals; original detector states unchanged.
Peakreserved6.8242GiB(control)/7.3320GiB(joint). NativePID167167 terminal.
Engineering weights are excluded from formal training. Evidencecommit1ffa084.

Implementation9ec24db plus root fix7fabcd8; exact CPUgate003. All six executable
source roots remained fixed through complete formal closure. The complete local
verifier passed. Source freeze is released; retain sealed F8 code/results and
use separate versioned experiments for future interventions.

Scope: single-seed internal fold with author-pretraining overlap. This supports
encoder optimization scope only; no geometry allocation, main-validation,
cross-seed or fading conclusion.
