# F9: complete correspondence-coupled coding result

The prelocked primary geometry contrast is not supported in this experiment.
AtAWGN10 the true correspondence armP is0.1068979percentage points below the
same-parameter generic armG and0.3642168below the shuffled armS. P exceeds
uniformU by0.8810601, which cannot distinguish geometry from head capacity,
mixing or optimization effects. Do not replace the primary condition after
observing another SNR, select a different checkpoint or call this a confirmed
geometry communication gain. This single-seed result also does not establish
that all geometry mechanisms fail or that these small differences are significant.

| Arm | Identity | AWGN6 | AWGN10 | AWGN18 |
|---|---:|---:|---:|---:|
| U, continued uniform |31.863902|25.307348|28.552510|31.716038|
| G, generic shared head |33.249208|26.728882|29.540468|32.187812|
| P, true epipolar coupling |33.207239|28.190435|29.433570|31.705206|
| S, shuffled coupling |34.531608|27.156091|29.797787|33.489575|

CarModerate3D AP_R40percent, IoU0.7, complete fixed internal372frames.
FullEasy/Moderate/Hard for all16endpoints is in formal-results-001.csv and the
terminal/local proof. P−G atAWGN6 is positive, while identity/AWGN10/AWGN18
are lower; this is a descriptive secondary observation, not a replacement
confirmatory result. No uncertainty interval or independent training seed was
computed. Author detector pretraining includes this internal fold; these are
not results on the complete public3769-frame validation set.

## Complete execution and evidence

Protocold8b3532 preceded implementation and formal launch47491f2. All four arms
strictly initialize the same539-state model from the finalF8joint parent,
with an identical new zero-output head. Each executes3340 actual paired native
AdamW updates; all13360raw rows and62empty-GT frames per arm are retained.
U trains51parameter tensors/fixes488; G/P/S train55/fix484. Each attempt uses
62400complex symbols atEs1, including49920stereo and12480appearance, with no
new sender-private receiver side information. G/P/S have the same561head
parameters. P/S incur additional correspondence compute; equal training
updates and communication resources do not imply equal compute.

All56external commands completed:4training,4native training audits and
16×inference/officialAP/feature audits. Every final endpoint covers372frames
with539readonly states and full dedicated CUDA noise-generator replay in the
native audit. The original21files and eight frozen source trees match at
closure. PID260096 actually exited before terminal artifact checks. Native
saved checkpoints/initializations, all5952prediction texts, calibration/GT,
metrics/pickles/communication records and final feature records were rehashed.
Local independent verifier002 checks all6156transferred artifacts, all13360
paired raw training rows,5952inference records and5952prediction texts; all
prelocked differences independently recompute exactly. CUDA RNG, actual
saved weights and officialAP were replayed/audited on sheng, not locally.

The prepared001closure/localverifier assumed all AP files lived below the
checkout. Actual manifests place them onD:. Neither001program was executed;
path-only clarification1e92691 preserved them and introduced002 with explicit
native-to-local transfer mappings. No running experiment, source tree,
condition, metric or selection changed. The correct-path complete closure
and verifier both pass.

## Exploratory recorded-gain observation

After completeAP closure, a descriptive analysis of the final500training rows
(steps2841–3340, not a selected AP checkpoint) finds mean absolute raw gain
deviation from1 ofG0.008681/P0.006762/S0.006361. Median within-frame gain spans
are0.018086/0.019116/0.009211. This describes relatively small learned gain
changes. It does not explain the AP differences causally, quantify the
counterfactual benefit of allocation, establish calibrated uncertainty or
support increasing the gain range/learning rate without a new protocol.

## Following work

Retain this complete negative primary test. Finish original full main source
inputs and matched detector baselines first, keeping all154scenario requirements
visible. Before a new geometry intervention, distinguish an actual geometric
representation improvement from generic learned reweighting and quantify an
operational benefit under matched resource/exposure controls. Main evaluation,
fullSNR/fading curves, multiple training seeds, new mechanism evidence and
manuscript/submission remain required; this result does not complete the project.
