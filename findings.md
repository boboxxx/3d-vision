# Current findings

Latest verified status (2026-10-04): both complete clean detector references
finished all3769 validation frames and passed independent all-file/AP audits.
Car moderate3D AP_R40 IoU0.7: Stereo-RCNN34.0749706%, LIGA67.7238123%.
These are different detector references, not a communication gain.
All KITTI archives, pairing/calibration checks and upstream train/val infos
are complete. Historical test counts below refer to their original stages.

F1 feature-only pretraining completed five epochs/16700 updates. Independent
checkpoint/scalar/holdout audits pass: all484 original tensors exactly unchanged,
all35 student optimizer states at their expected update counts. Epoch5 holdout
normalized MSE left/right stereo/appearance=.324682/.326992/.665291. Strict
uncompressed372-frame prediction/raw-GT/native-AP audit passes: Car3D AP_R40
E/M/H=20.713529/11.002872/9.640570%, Pedestrian10.613168/8.256384/6.961676%,
Cyclist0%. This restores some detection relative to F0 uncompressed0%, but is
far below the99.862902% clean-author moderate diagnostic on the same fold.
It is not communication performance or a main-validation method gain. Fixed
F1 sources were rechecked at closure before subsequent source changes.

F2 completed the separately locked3340-update native task epoch from F1,
student-only35 parameter tensors, original receiver parameters/BN frozen.
Full-state/optimizer/CRC audit passes; only nativeglobalstep and2 imitation
normalization buffers change. Strict372-frame independently recomputed Car3D
AP_R40 E/M/H=65.754773/42.319460/34.586988%, Pedestrian23.729840/21.056252/
19.075320%, Cyclist3.853234/3.842723/3.842723%. Both AP and saved-state evidence
pass. Initial audit's wrong LR index set was repaired outside frozen training
sources; native before/after logging covers0..3340 while updates are3340.
Automatic versus fresh inference has same1493 predictions/classes, with small
AP differences (Car moderate42.315798 versus42.319460); both are preserved,
unique cause unestablished. Second fresh inference independently audits at
42.317060% moderate, maxAPdifference across metrics .015269 percentage points;
no bitwise identity claim. No main-val or
wireless result. Use strict predetermined AP, never select the best rerun.

F3 completed one fixed3340-update uniformAWGN epoch, original519 state
scope frozen except3 allowed native buffers. All28 link optimizer states,
CRC scalars, strict372-frame AP/channel energy and all4 source trees audit.
Car3D AP_R40 E/M/H at10dB=.261517/.063488/.038805%, Pedestrian/Cyclist0.
This is a negative exploratory result, not useful communication performance.
Keeping the same full codec and weights but removing noise also fails:
identity-channel Car3D AP_R40 E/M/H=.137079/.039805/.016736%. No unique
architecture or optimization failure cause is established by these AP results.

Complete passive372-frame identity feature diagnosis also audits. Cost NMSE /
cosine=1.233875/.108073, appearance=1.065629/.034929. At the same strides,
plain pool/interpolate reference gives cost=.592454/.639105 and appearance=
.127549/.934426. These are reconstruction references, not optimal information
bounds or AP ceilings. Current decoder features are poorly aligned even
without noise. A direct feature-reconstruction warmup is therefore a better
next controlled intervention than comparing power allocation on a failed codec.

F4 protocol locked in3eb2d9f; trainer/auditor/evaluation implementation in
f94205b. Initialize all549 finalF3 states exactly, update only16 codec tensors,
keep all533 other states (including nativeglobalstep and imitation buffers)
bitwise unchanged. Fixed one3340-update identity epoch, costMSE+appearanceMSE,
no teacher or task forward. Three-update native GPU engineering and independent state/scalar audit pass:
all16 Adam counters=3, all533 inactive states exactly fixed; student6 calls,
codec3, forbidden teacher/downstream0. No formalF4 result exists yet. Full
identity and10dB native AP/channel/feature audits are queued after training.

The isolated original reproduction variant includes actual pretrained SpyNet,
full30-block residual stacks, sensor-only YOLO caches for3340/372 paired folds,
TableIX channel CNNs, sparse transmitted key cells, serialized repetition/CRC
box control and pilot-only Rayleigh estimation. Local/shengCPU718-gradient
checks pass. Native370x1224 GPU stage1 and finaljoint-stage5 sanity passes,
562/718 active gradients finite and frozen states unchanged; weights discarded.
Five-stage controller implements all83 specified epochs and8 phase-boundary
CPU optimizer probes pass. The native RGB loader, full staged trainer/auditor
and formal original communication training/AP remain required, not completed.
This is a documented variant, not located author-code or reproduced published AP.

First codec epoch completed3340 updates. Complete scalar and frozen-state
checkpoint audits pass:481 original tensors unchanged, only globalstep and
two training-only imitation scales differ,63 new tensors have actual optimizer
updates. Native10dB holdout Car moderate3D AP_R40 is0.0076202%, so optimization
has not yielded useful detection. Resource-file audit rejects empty DDP logs;
fixed recording uses returned sensor metadata. Fresh explicitseed17
evaluation independently audits at0.0099900% moderate AP, all372 frames
with62400 complex uses and matching energy. Original negative outputs remain preserved.

D0 clean detector on the same372 codec holdout independently audits at99.8629021%
Car moderate3D AP_R40. This high value reflects a holdout included in the author
checkpoint's original detector training; it is not the main validation result.
The trained student without communication independently audits at0.0% Car
moderate3D AP_R40: sender features are not usable by the frozen receiver. Allocation
comparison is deferred until useful sender/codec features are established.
No scientific method gain is supported. Protocols and rationale:
experiments/geometry-link/{tuning,diagnostics}.md.

The sections below preserve findings at their original stages. Statements about
pending downloads/clean AP or unimplemented modules are historical; current
results are documented above and in docs/liga-baseline.md.

## Established from sources

The original paper reconstructs stereo RGB for a separate pretrained detector.
It preserves detector compatibility but does not optimize the communication
link jointly with detection. LIGA-Stereo provides a full stereo-to-3D pipeline
with stereo geometry features and direct task losses. Its released environment
is old and must be validated on the actual GPU before choosing dependencies.

## Working hypothesis

Communication errors should be priced by their expected geometric consequence.
For rectified equal-principal-point stereo, Z=fB/d; small disparity errors give
Var(Z) approximately (fB/d²)² Var(d). Entropy alone is insufficient: identical
disparity entropy can imply different depth errors. For a multimodal posterior,
compute depth moments directly rather than applying a local approximation.

Start with a complete geometry feature link and a constrained allocation
rule. Keep token selection and digital variable-rate coding for later evidence
driven decisions. Sparse selection introduces position signaling and can make
purported communication savings disappear.

## Unresolved

No original communication implementation was located in the searches performed
so far. This is not evidence that no implementation exists. Original full
reproduction remains required. A branch-matched clean Stereo-RCNN reference is now verified; no trained
communication-method result exists yet. The new method's
novelty, calibrated uncertainty and edge compute cost remain to be tested.
ICC 2027 extension/track availability is unverified.

## Implementation evidence

Built geometry/appearance codecs, exact posterior depth moments, constrained
Gaussian-proxy power allocation, unit-energy complex AWGN, and noisy-pilot
Rayleigh decoding with explicit energy/channel-use accounting. Patched the
pinned full upstream detector to use both decoded streams before downstream
stereo/appearance heads. Added task-loss posterior supervision, a seeded full
detector launcher, channel measurement JSONL and dataset split/pair audits.

Nine local engineering tests passed: noise convention, constrained allocation,
pilot accounting, depth moments, task gradients, absence of a clean residual,
absence of inference GT dependence, full config preservation, patch idempotence
and split-overlap rejection (some checks are grouped in one test). These are
engineering checks; they establish no detection accuracy or research hypothesis.
Full GPU engineering integration now passes on sheng's RTX 4090. All upstream
CUDA extensions and the complete author model run in the isolated project venv.
Strict loading verifies all 484 detector/teacher tensors. Full-resolution clean
and random-codec synthetic inference and the complete training-loss backward
pass succeeded. These inputs are synthetic; no KITTI AP result exists.

Pooling strides (8,4,4) for geometry and 4 for appearance reduce symbol count
before learned coding. Reconstruction quality and geometry preservation must
be measured; a low symbol count by itself establishes no task benefit.

Runtime audit found two substantive upstream details: the released image crop
is 320 pixels high and detector range is 2–59.6 m. The protocol now follows
these settings rather than assuming full original image height or 80 m range.

An inference audit found the released code still executes its training-only
LiDAR teacher and requires GT depth for diagnostics. Teacher outputs are used
only by training imitation loss; skipped its evaluation forward and made depth
diagnostics conditional on GT availability. Model structure/checkpoint contents
stay complete. The sensor-only synthetic pass verifies zero teacher calls.

Independent numerical checks validate cost-volume interpolation and gradients,
spconv kernel-layout conversion against dense convolution, voxel coordinate
order, and analytic IoU/NMS examples. The unchanged KITTI evaluator also passed
its independent fixture after enabling NVIDIA CUDA Python bindings; Numba's
ctypes interface crashed with a live PyTorch context in WSL. Complete data
download, infos generation and clean AP verification remain the next gates.

## New measured constraint and evaluation gate

All eleven local engineering tests pass. Matched RGB and earlier raw-cost
geometry links pass full GPU forward and task gradients, without training.
Dense sender compute lower bounds are 922.27 GFLOPs (processed geometry),
619.01 GFLOPs (raw geometry) and 15.37 GFLOPs (RGB). Geometry is currently
compute-heavy; see docs/compute-audit.md. A useful uncertainty allocation alone
would not establish an edge deployment advantage. Task sensitivity and student
encoding remain unimplemented candidates requiring evidence.

All full KITTI stereo inputs are now verified. The full author Stereo-RCNN
clean baseline is running on all 3769 validation frames. Original postprocessing
and KITTI serialization match the author demo/writer; the unchanged evaluator
passes both strict and relaxed synthetic fixtures. Real AP remains pending.
Velodyne still downloads for full LIGA data preparation and clean validation.

Source audit subsequently found different official weights for master and
branch1: 139 common state tensors differ, and branch1 supplies 104 BN counters.
The first full run used master weights with branch1 code; it is reclassified
as compatibility evidence with a separate interpretation file. Branch-matched
clean validation will use the newly downloaded epoch13 branch1 checkpoint.
Full tensor shape agreement did not establish source/checkpoint equivalence.

## Completed clean reference and exploratory student

Branch-matched Stereo-RCNN clean validation finished all 3769 frames. Car 3D
AP_R40 at IoU .7 is 53.1215/34.0750/27.8535% Easy/Moderate/Hard. Independent
all-file audit verifies prediction hashes/counts, finite KITTI rows, fresh
label/calibration identities and unchanged-source run metrics. This is a clean
detector reference, not original communication reproduction or method gain.

All five KITTI archives now have final SHA-256 and 7481-file CRC checks; upstream
LIGA train/val preparation is running. Optional student sender and nonlinear
raw posterior pass full GPU forward/backward; all 55 student/codec parameter
tensors have finite task gradients, all original 484 tensors load. Feature
and LiDAR teachers never run at inference. Tx dense lower bound 12.48 G and
stream time 15.99 ms remain engineering measurements of untrained weights.
Linear raw posterior was structurally unable to use depth-constant left input;
corrected nonlinear head can. Earlier linear measurements/source are retained.
Task-sensitivity prediction is now implemented and GPU-verified, but no
calibrated posterior or trained AP gain exists.

Native sensitivity supervision isolates the original 3D classification/box/
direction objective before in-place imitation additions. Independent analytic
check excludes a deliberately overwhelming imitation loss and confirms no
second derivative changes the received-symbol gradient. Inference is unchanged
when GT depth is omitted. Full GPU check passes all 63 new parameter tensors
with finite gradients, no optimizer steps. Profile Tx12.51 G lower bound,
17.11 ms stream time. Sensitivity times variance is an empirical allocation
heuristic requiring matched-capacity/extra-supervision ablations and actual AP.


F4 completed fixed3340 native updates from F3, checkpoint984771355f145441f785a8bca5ddde2762fec4925d504298b7fb9abe8c3f57f0. Independent audit: all16 Adam states3340, all533 inactive states bitwise fixed,3340unique returned frames, student6680/codec3340/forbidden0calls. Training first/last100 median loss1.1494748592→.6214553714.
Two predetermined full372 native evaluations and AP/channel/feature audits passed: Car3DAP_R40 all difficulties0% under identity andAWGN10dB. Identity cost/appNMSE .6102321605/.1500258911, cosine .6243167767/.9219620784; AWGNcost/appNMSE .6182211176/.1671246815, cosine .6180713901/.9128078680. Feature reconstruction substantially better thanF3, yet detection unusable. No resource-allocation/communication gain claim. Four source trees identical at closure. Raw scalars/features, manifests and independent audits retained. Launch metadata commit-label typo corrected transparently; both records retained and experiment unchanged. Next investigate task adaptation/geometry pooling with separately locked protocol and complete original staged baseline.

Original staged trainer3 real native GPU updates passed independent complete audit:562 active parameter tensors updated and changed,206 inactive states exact fixed, complete768-state architecture/freshseed17+officialSpyNet initialization verified, realAdam step3 for all562. Sensor images read with cachedSHA verification. Engineering checkpoint58288281c16f54301d04ccf5053e829c86522666ced4d912e4678eff41e2b4d8 is not formal initialization. Implementation/protocol precommittedbfee9d8; isolated local/server sources allmatch. Launch fixed original83-epoch cycle only after this evidence is committed.

Formal original variant83-epoch cycle launched PID26642, precommitted implementationbfee9d8 and engineering evidencecea8af3. Stage1 live actual300 updates confirmed; no completed-stage/AP claim. All isolated reproduction/cao2025 executable/upstream sources and protocol frozen for the full cycle. Next original stage only after complete independent predecessor audit. Large weights/snapshots stayD. F4 follow-up five-condition pooling diagnosis protocol locked for later GPU scheduling, no training or physical communication claim.

Pooling diagnosis nativeGPU engineering passed all5fixedconditions onheldoutframe000036, preprocessed320x1248, complete519states matching fixedF4 and unchanged; exact30 unused link states verified, student2/cost1/teacher0. Peakreserved3.845703GiB. OneframeengineeringnotfullAPclaim. CurrentGPUcoexistence gate andunchangedliveoriginal21sourcefilemap verified. Formalall5×372 inference pending; implementation65a3253 andprotocola88a62e precommitted.

Originalstage1first2epochs completedand independentlyaudited: full768savedstates,206 inactive incl60flowparameters exactinitial,562Adamstates3340/6680 actualsteps,6680sealed raw prefix rows with full3340coverage eachepoch. Whole12epochstage incomplete; livephase3 flowunfreeze reached7000updates. Poolingsuite002 PID27918 actualframes nowrunningafter nativeNCCL/DDPprobe002; first001 zero-framefailure preserved, no protocol/checkpoint/conditionchange. Rootcode/sourcefrozen until suite terminal closure; originalisolatedsource remainsfixed.

Complete prelocked5condition pooling suite002 closed all1860nativeframes, each AP/GT/predictions,519state, operation metadata andall4source trees audited. Car3DModerate AP_R40percent: {"control": 42.31409807404212, "cost_only": 0.016722408026755852, "appearance_only": 40.39588128173474, "both": 0.0, "depth_preserved": 0.07899628252788103}. Rawcost compression alone damages task severely; appearance is comparatively robust. Depth-preserving spatialpool also nearzero, rejecting the simple explanation that depth averaging alone is enough to explain failure. Shift next exploratory boundary to stereo feature transport andreceiver geometry construction, preservehorizontal epipolar resolution at identical62400symbols; no guaranteedsuccess/novelty orwirelessgain.

F5b corrected full-resolution stereo-feature warmup completed all3340 updates and predetermined identity/AWGN10dB full372 evaluations. Full535 states loaded; all519 F2 states exact unchanged; all16 Adam states3340 and all16 learned. Native receiver builds cost strictly after counted channel, received features only, teacher0; all535 states readonly in eval. Both AP/GT/prediction/energy/passive feature reductions/source audits passed. Identity Car3D E/M/H1.8477074202/1.1551948753/.7903439153%; AWGN .5367540029/.1333333333/.1116625310%. Full4 sourceclosure consistentlocal/server, original21 files fixed. Initial wrong-interface F5 failure preserved and independently verifiedzero updates/no learned checkpoint; revised protocol1494f2c precededimplementation34a40ab, evidence9926412 precededformalPID30987. Final checkpoint9a4b291070df160e6c03e75ed31eeee1dda7f6bfbaa0528a43a9cdf526977553. Identity stereoL/R/appNMSE .5927285539/.5881644121/.1347826783; AWGN .6171852779/.6128138578/.1424511104. Reconstruction lossfirst/last100 medians1.2475757003→.6131930649. Minor task recovery does not establish usable model or superiority versus F4/other boundaries, whose histories/layouts differ. Next require separatelylocked clean3D task compatibility and laterwireless adaptation before any allocation study.

F6 firstformal001 failedafter5updates due to wrapperrejecting nativeemptyGT, nofinalcheckpoint/AP. Failurefullsourceclosure preserved before repair. Native dataset resampling precedes class/rangefilter and canreturn[1,0,8]; native anchorassigner supportsbackground-onlyloss. Separatelylockedrepairde002cd andimplementation950067f accept thiswithoutskip/dummyGT/datachanges. Repairedsix-stepGPUengineering independentlyvalidatesactualAdamstep6/all16changed,519fixed,emptyframe000026/background525312/zero regressions,finitefullreceiver/channelgradients andreadonlyDDP535. EngineeringdoesnotestablishAPimprovement; freshformal002 retainssoleF5b initialization andsame3340/two372protocol.

F6b completedandfullyclosed: frozen519receiver/student states with16codec-only native3D adaptation reached CarModerate25.39930518% identity and18.51584646% AWGN10 onfixed372, vsF5b1.15519488/.13333333. Same62400/layout, extra3340GTupdates explicitlycharged, no equal-exposure superiorityclaim. All535/actualAdam/frozen/rawscalar/nativeGT/energy/prediction/calibration/readonly/source audits passed. Reference-featureNMSE worsened(stereo~.70,app2.03 identity) whileAPimproved; featureMSEalone doesnotorder task outcomes inthispipeline. Thisis empirical taskcompatibility recovery, notuniquegeometrymechanism ornovelty proof. Identity→AWGN gap6.88346pp motivates a separatelymatched extra-clean/noisy adaptation test. Mainval, calibratedpreventabledamage, allocationcontrols, fading/pilots, trainedmatchedoriginalbaseline, seedsandpaper remainopen.

2026-10-04 Originalfullstage1(12epochs40080updates) andstage2(10epochs33400updates) complete independent full768-state/everycheckpoint/actualAdam audits passed. Local sealed73480trainingrows and2x372lossdiagnostic rows verified against manifests/audit SHA, native coverage/order/finiteness/source21/protocol andwholeparentchain. Stage1 final562nonflowAdam40080/60flow33400,146inactivefixed; stage2 16actualAdam33400/752inactivefixed. Stage3 latest4600actualupdates, PID26642live; full83epoch/trainedwireless3DAP notcomplete. Original21code staysfrozen. NativeauthorLIGA RGB threeGPU engineering endpoints passed(cleanrelay/fresh768CPUcodec identity/AWGN10),484detector+768codecreadonly, receivedsensor-only exactcrop/calibration, nativeimagebackbone2/neck2/cost1/head3D1/forbidden0. Peakreserved4.09765625GiB andphysicalmarginpassed; noAP/finaltrainedbaseline claim. Sources/protocol/counters verified aftersync.

F7 matched-noise follow-up has a complete isolated implementation and13meaningfulCPUregressions, but no nativeGPU orAP evidence. Equalexposure claim will require bothsameparent/full519frozen/16Adam3340 andall3340 actualaugmentedinput fingerprints matching, independentlyreplayed dedicatednoise state andfourpredetermined372AP endpoints. Noise RNG mustnotconsume model/data globals; callerprovidedRNG/mismatchedSNR/layout andGT/private inputs are rejected. Root/upstream/originaltrainer source identities stayunchanged, so pendingdetectorprobe/currentoriginalstages retaintheirsourcevalidity. Do notinfer APimprovement ordata-matching fromCPUfixtures.

Original stages3and4 now independently complete/audited (6epochs20040 fusion-onlyupdates,10epochs33400 allsemanticupdates); local sealed53440records/order/coverage/wholeparentchain verified. Stage5 is stilltraining, not a finalwirelessbaseline. F7 deployedsourceidentities and13meaningful CPUchecks passedsheng after syncingmissingoldtestfixture; nativeprobe/engineering remainqueuedbehindphysicalGPUmargin. No newtrainedAP, allocationgain, ornovelty claim follows.

Final trainedoriginal-codec AP workflow nowimplemented/deployed withtwo losslessreceivedcaches andsix predeterminednativeconditions, exactpairedreceivedinput/erasuredenominator/resource/actualnoiseaudits, full535-style readonly checks adaptedtoauthor670/484andcodec768, independentfreshKITTI/AP andterminalsourceclosure. CPU5contractcheckspasslocal/sheng; newreceiverGPUengineering queuedafterF7engineeringclose. These checksestablish implementationcontracts only, not trainedAP/fairsuperiority. Solefinalfull83-epochstage5epoch45 auditedweights andnativeengineering are hardgates.


The original-reference scenario audit changes the main experiment scope: source
compression10/30/50, AWGN6–18 integer and ideal-CSI exact-ZF Rayleigh6,8,…,18,
both actual digital LDPC/QAM configurations, author's StereoRCNN and primary
Car IoU.5 R11/R40. Our noisy-pilot/internal372/.7R40 runs are explicitly separate.
The fixed grid has154 unique conditions (167 table appearances), not154 results.
Seven CPU checks and a single training-frame JPEG/JP2 stream audit passed on
both environments. Native JP2 rates10/30/50 measured10.00265/30.01027/49.984×
rawRGB8; all framing bytes are counted. No LDPC coder/digital AP exists yet.
ECSIC official pinned release reports entropy estimates, not serialized streams;
using these as physical source bits would invalidate a digital comparison.

F7 first native engineering failed before any optimizer update because CUDA
generator cuda had no explicit index while the tensor was cuda:0. Full535
initial states match F6b, with zero rows/checkpoints; independent closure retained.
The separately prelocked repair resolves only the alias to the current index.
Nineteen CPU checks pass; recoveryqueue002 now waits for original actualterminal
and whole stage5 audit before fresh engineering002 and both final GPU receivers.
This failure measures no AP/noise sensitivity and supplies no evidence of OOM.

Artemis now supplies independently verified RTX PRO6000 Blackwell97887MiB
compute. A dedicated torch2.7.1+cu128 environment passed actualsm120 matrix/
conv3d forward/backward and frozen original21-source native RGB stage1/stage5
checks:562/718 finite active gradients,206/2 frozen states unchanged. Two
untrained discarded updates establish compatibility only; not throughput/AP.
Four Slurm jobs completed exit0. CPU-only complete KITTI deployment11423915
is submitted; full dataset and native detector operators remain unverified on
Artemis. The project still requires same-detector/rate/energy/exposure controls,
preventable-damage calibration, geometry/allocation evidence, mainval and seeds.


The digital baseline gap is now partly resolved by an actual pinnedNR38.212
encoder/GrayQAM/APP/BP chain, not a code-rate simulation. Complete native
training-frame JP2 stream passes zero-noise bytes/CRC for both reference rates.
At10dB the fixed synthetic packets have real errors, retained without fallback;
this does not estimate a useful BER/noisy image AP. Actual whole-block padding
produces181116 vs181278 uses for the same90545bytes despite both nominal4
bits/use. Packet population-Es1 normalization yields actual energies180816.57
and181008.78 rather than exactly symbol counts. No private per-packet scale is
communicated for free. These differences make serialized bits, signaling and
realized energy essential to the later comparison. Pinned code-family/block/
iterations are explicitly chosen variants because author settings remain unknown.


The full declared original reconstruction variant is now a trained baseline:
83epochs/277220 real updates,allfive full-state/Adam/raw-record audits and
actual terminal process proof pass. Final full768 state checkpoint08a7c848 is
fixed,not selected by holdout. Local sealed stage5 audit verifies150300 raw
rows/actualupdates and25/20 phase split with charged control/data resources.
This establishes baseline training, not exact author reproduction or AP.
Repaired F7 six-step pair andbothfinal RGB receiver GPUinterfaces close with
full535/670/484/768-state andpaired actualinput evidence. The CUDA alias fix
now has native evidence; no favorable detection claim follows from six updates.
Next formalF7 controls extra3340 training exposure fromthe sameF6b parent.
Original final internal372RGB evaluation noweligible with samecached decoded
inputs acrossbothdetectors; rates/exposure remain unmatched to new62400 method.


Artemis fullKITTI nowverified andCPUdeploymentterminal: fivearchives wholeCRC/
SHA/member identities,all7481 componentIDsets andfixedtrain3712/val3769 SHA
matchsheng. Bothcomputehosts have identical public data; no newsource labels
orvalidationresults were inspected duringdeployment. DetectorCUDAoperator
portability onBlackwell remainsunverified, despite originalRGBmodel anddigital
chain execution. Next useits availableGPU onlyfor separatelylocked substantive
workloads; do notreserveidleGPU orsilentlyreuseCUDA11.8 sm89 binaries.


### Equal-budget channel adaptation closes with a useful internal effect

F7 formal seed17-001 completed both3340-update arms and allfour372-frame endpoints, whole535-state/519frozen/16actual-Adam audits, matched actual augmented inputs and independent closure. Both cycle/wrapper actualterminal; all46 artifacts and6680 raw training rows retained and locally verified. At fixed62400uses/Es1, identity continuation has Moderate3D R40 IoU.7 AP25.60749596(identity)/17.40286363(AWGN10); AWGN-adapted weights26.19152328/24.53278277. Prelocked AWGN primary delta+7.12991914pp. The controlled added-update comparison shows channel exposure matters beyond adding another epoch. It does not validate uncertainty, allocation, mainval, fading or cross-seed significance. Internal author-pretraining overlap remains. Keep the separately prelocked geometry pilot on F6b; future allocation baselines need matched channel adaptation/exposure rather than the older identity-trained baseline.

Artemis native operator toolchain progressed: official CUDA12.8.1 components checksum-locked, standalone unchanged cost-volume/IoU-NMS/ROI sources compile to sm120 in CPUjob11423973 COMPLETED0:0, runtime package freeze unchanged. Actual GPUprobe11423980 is pendingQOSGrpGRES. Build alone does not prove GPU/full-detector compatibility. Main remaining science is whether sensor-only geometry predicts preventable communication task damage, followed by actual matched-resource allocation AP.


### Native geometry-risk diagnosis is engineered, not yet calibrated

Eight CPU checks now pass on both local and sheng runtimes with exact current code/metadata hashes. The clean gradient is taken with respect to normalized transmitted symbol leaves through a frozen receiver; masked CN(0,.1) interventions retain every symbol use and actual draw. Padding support is derived only from public crop dimensions at both stereo correspondence endpoints. These checks establish diagnostic plumbing and avoid confusing first-order loss variance with expected positive task damage. They do not establish that geometry uncertainty predicts preventable damage. The two-frame260-pass native engineering queue waits for the running original RGB cycle and its closure; full64 calibration has not run. Independent replay of complete native inputs, public calibration, all noise and depth posteriors is prepared before observations.

Both fixed trained-original RGB caches now contain372 frames and passed server cache audits; native clean relay evaluation has started. Detection results remain pending, and unequal original/new method channel resources and training exposure still prevent a fair superiority claim. Artemis operator GPU allocation remains pending; CPU compilation cannot establish full native detector compatibility.


### Digital reception no longer needs an oracle byte count, but decoder identity matters

The concrete20-byte P6SBheader now traverses the realLDPC/QAM/BPchain and is counted withpayload/padding. Receiver extracts sourcebytes fromdecodedheader only; capacity/padding/CRC/codec failures erase thewholepair. Bothnativeidentity configurations recover the exact90545bytes. Ten engineeringpackets/sixsuccesses/fourheadererasures and all18rawarrays/streams pass independent bit/noise/APP/accounting audit. This closes the receiver-length plumbing gap for this declaredvariant, notmainnoisyKITTIperformance orphysicalsyncsignaling.

Independent local re-decode exposed nativeJPEG2000 pixel differences acrossplatforms despite exactreceivedbytes:48left/42rightuint8values differby1. An independentlyparsed/decode original-runtimeCPUreplay exactlymatchesall6originalsuccessfulpixelrecords andcompletearrays. Retainfirststrictauditfailure; crossplatformpixelidentity is notclaimed. Sameauditeddecoder orsharedlosslessreceivedpixelcache isneeded forfairAPcomparison. The exact cause ofrounding differences remainsopen; do not blame channel noise or change a measurement tolerance afterseeing it.


### Trained RGB baseline works; remaining question is matched resources and geometry damage

The complete83epoch originaldeclaredvariant now has auditedtrained detectionresults onbothauthorreceivers. SameactualreceivedfloatRGB gives StereoRCNN Moderate40.49/40.53/41.76 and LIGA99.86/79.36/75.92 underclean/identity/AWGN10. LIGAidentity reconstruction incurs20.50pp loss andnoisyreconstruction23.94pp, whileStereoRCNNdoesnotfollowthatorderinginthissingle-seedinternalfold. Never inferbeneficialnoise orreceiver-independenttaskdamage fromthatpattern. Both use200065meancomplexuses, versus62400newdirect3D; traininganddetectorpretrainingexposure differ. A frontier/operatingpoint comparison isrequiredbefore superiorityclaim.

Native geometry engineering closes260passes/all535states/completeindependentfullarrayaudit, so the proposedriskdiagnostic can now be tested. Bothengineeringfullnoise samples slightlyreduce native loss; cleanambiguity alone cannot be assumed to implypositivecommunicationdamage. Native gradients diagnosefirstorderlossvariance, notmeanlossorAP. Full64auxiliaryfit/out-of-fitprotocol isprelocked, withdistinct continuednoise and explicitfinite-sample positivepart target. Engineeringobservations are excludedfromfit; repeatedfirsttwo scenes aredisclosed. F7channel-adaptation controls remainrequiredforanylaterallocationcomparison. Mainval/fading/seeds/novelty/paper arestillopen.

### Full64 risk pilot separates weak mean-damage evidence from variability

All64 fixed training scenes/8320 nativepasses/519168000uses closed with535
readonly states/no optimizer; complete independent native noise/input/geometry
replay passes onsheng. Two auditor-reference failures are retained, repairs
prelocked at unchanged tolerance and without altering observations. Full8320
raw metadata and allfive ridge fits/ranks/1000 frame-bootstrap outputs independently
recomputed locally; all2048 common-support groups defined, no missing frames.

Geometry-only positive-target Spearman.04455,95%CI[-.02273,.11324], energy.10553
[.04209,.16852], shuffled.02832[-.02847,.09208]. Combined.08814 and MSE.000440612
versus energy.000443839; this small point difference is not a tested reliable
increment. Current coarse stereo uncertainty lacks usable mean-damage evidence;
do not train a mean-risk allocation head on the unsupported premise.
Gradient correlation with sampled fluctuation variance.88341[.86156,.90165],
geometry.36986[.30214,.43053], shuffled.01005[-.06013,.07668]. This motivates
variance/curvature separation, not an AP gain. Positivepart of4-draw mean has
noise-variability clipping bias even when true linear expected damage is zero;
derivation in docs/four-draw-positive-damage-target.md. Native gradient is
label-dependent and training-only. Next fresh-scene paired±noise should separate
symmetric mean increments from antisymmetric variability before policy training.
All64 are perception-training scenes; no mainval/allocation/generalization result.

### Fresh8 paired noise reinforces mean versus variability separation

Complete8208 passes on8 new training scenes closed with independent native
noise/input/state replay, local full raw metadata checks and server/local frozen
predictor statistical recomputation. All256 regions common-support, all8 ranks
defined; no failures/refitting/exclusions. Geometry signed mean rank−.05984
[-.17852,.05280], variance rank.45120[.37458,.53418]; gradient signed mean.05091
[-.09898,.18997], variance.96014[.94804,.97242]. Currentgeometry lacks positive
mean-damage evidence. Ranking variability cannot justify mean-loss optimality
or AP allocation gains. Even strong gradient variance ordering does not validate
linear magnitudes: per-scene RMS(s−gTe)/RMS(s) ranges.6907–1.6731. Keep signed
increments, finite-pair MC uncertainty, labels-only gradient scope and small
training-scene sample limitations. Do not turn this into endless proxy studies.

F8 protocolad64d76, locked before fresh8 statistics, targets the stronger
uniform baseline: same fullF7AWGN parent, each3340updates,16codec control versus
51student+codec treatment, identical input/noise/resource exposure, four fixed
internal endpoints. Extra training and unfreezing are controls, not novelty.
Not implemented/launched yet. A later allocation policy must predict useful
resource-exchange benefit and pass actual matched-resource AP comparisons.

### Joint encoder adaptation is now native-engineered; formal comparison runs

F8 seven CPU checks pass on both runtimes. A prelaunch root-path error was
preserved and repaired before any GPU work. Both native six-update arms then
closed all535 state/actualAdam/gradient/input/noise audits and local complete
metadata verification: control16updated/519fixed, joint51updated/484fixed,
one legalemptyGT per arm. Joint peakreserved7.332GiB fits the physical margin.
This proves the intended optimization scope works, not that it improves AP.
FormalPID167748 is running from the same F7AWGN parent with3340updates each;
final four internal endpoints pending. Do not change frozen executable sources.

The additional primary-source overlap review reinforces that importance-based
power allocation, masking scores and ordered representations already exist
(ISFR, Generative Feature Imputing, SIAC). Specific comparisons and our own
matched AP evidence must carry a future stereo-specific claim. A variance
surrogate, encoder unfreezing or a renamed importance score is insufficient.


### ECSIC now has actual independently decoded source bytes

Recovered official Cityscapes lambda0.01 weights/config strict-match225states.
Two prelocked engineering pairs have causal four-stream receivers with E/HE
forbidden and all18arrays exactly matching the unchanged official forward.
A declared finite-CDF/rANS implementation now yields real813/44824-byte P6EC
containers, including166header/CRCbytes and all flushed states. Independent
compiled upstream C++ rANS reproduces every byte of all8native streams; full
state/array/source audits pass onsheng, transferred metadata/payloads locally.
One synthetic output-hash/read-barrier failure was retained and narrowly
prelocked/repaired by hashing serialized bytes in memory; no result tolerance
changed. Current evidence is fixed-runtime two-input engineering only.

The original comparator can now enter an actual LDPC/QAM chain, but the existing
P6SBv1 accepts two image streams and must gain a separate explicit joint-container
format. Complete received decoded blocks must determine length; sender-side
truncation is diagnostic only. Original10/30/50 operating points, KITTI adaptation,
physical noise/fading reception and AP remain open. These byte counts are not
a matched-rate accuracy result or a novelty contribution.

### Matched encoder adaptation closes with a modest internal AP improvement

F8 completes3340updates/arm, all535state/actualAdam/pairednoise-input audits and
four372frame endpoints. PID167748actuallyterminal; fullserverclosure and all46
transferredartifacts/6680rawrows independentlychecked locally. At62400uses/Es1,
codec identity/AWGNModerate25.664854/22.378536, joint28.013625/25.664906. Prelocked
primary+3.286369533pp. Morecodec-onlyupdates do not automaticallyimprove the
parentF7AWGN24.5328%; parentcomparisonhasunequalupdatebudget. Joint adaptation
helps the matched uniform baseline but does not establish geometry allocation,
mainval, crossseed or novelty. Extraencoderbackwardcompute isdisclosed; final
checkpoints only, noselection. All62emptyGT retained. Code/state freezeclosed;
retain sealedexperiment. Next scientificexperiment must test actualresource
exchange/policybenefit under matched physicalcontrol overhead andAP, rather
than fit another unsupported geometrymean-damage proxy.

### Actual ECSIC digital reception is now causally audited

All20 prelocked attempts traversed complete real entropy bytes plus charged
received-header framing and actual NR-LDPC/QAM. Independent full PHY replay closes
11 receptions/9 whole-pair erasures; native Rayleigh18+256QAM fails decoded-padding
validation and is retained without fallback. This establishes engineering of the
actual comparator chain, not AP or packet-reliability statistics. Next neural
reconstruction must use each actual accepted packet on the same fixed sheng CPU
runtime, with public header crop dimensions and no clean-image fallback.

### Test correspondence through task AP, with capacity and geometry controls

F9 prelocks Uuniform, Ggeneric shared head, Ptrue epipolar coupling, Sshuffled
coupling. Same535-state F8joint parent; same new four-state/561-parameter head,
539 full states, matched3340 updates and62400 symbols. Receiver uses received
symbols and public layout only; gains change the learned representation and are
not inverse-divided. This is geometry-conditioned JSCC, not proven pure UEP,
uncertainty calibration or novelty. Core and training CPU gates pass; native
six-step engineering/full state audit precedes formal16 internal endpoints.
PrimaryP−G AWGN10Moderate and requiredP−S directly test geometry rather than
reusing unsupported mean-risk proxies. Main validation/scenarios/multiseeds remain.

F9 now passes six native updates in each arm and full539-state/Adam/input-noise/
receiver chronology closure. Allfour initial loss/actual energy records agree;
each retains one emptyGT; all24 raw rows and27 transferred artifacts pass local
verification. Original484 detector states remain fixed; U also fixes the new
four-state head. This resolves implementation/gradient feasibility without
changing the scientific status: no F9AP is measured yet. Preserve core source
identities while adding evaluation integration, then complete the prelocked
four-arm16-endpoint comparison before interpreting geometric benefit.

The native F9 inference interface now closes48 fixed engineering frames. Its
explicit3D path gives exactly the same boxes/scores/classes as a reference
including unchanged auxiliary2D/depth heads, using the very same received prefix
and no additional noise, at all eight first-frame conditions. All539 detector
states remain fixed; independent full RNG advancement and raw prediction audits
and all96 transferred artifact checks pass. This resolves an auxiliary-execution
compatibility issue before formal observations. It gives noAP benefit evidence;
four-arm task evaluation is the next scientific test.

The prelocked F9 scientific comparison is now running from the sole F8joint
parent. All four arms have the same3340-update schedule/resource budget and final
checkpoint rule. Independent final auditors preserve the original prediction/GT/
official evaluator and add strict539/publicarm/SNR/coding/RNG checks. All16
endpoints must close before interpretation; partial training is notAP evidence.
The active source freeze must not be disturbed while advancing unrelated ECSIC
reception work outside those eight source roots.

### Received bytes now close the ECSIC neural receiver boundary

All11 accepted actual digital packets causally decode on the sealed CPU runtime;
9erasures get no neural output. Independent server verification finds exact
198full arrays/22native FP32 crops and all225states fixed. Terminal and local
whole-metadata/22received-byte checks also pass. This removes the causal
reception/cropping gap; it does not resolve domain adaptation, original rate
points, reliability statistics or detection AP. One fixed source per size and
unchanged accepted containers make equality a compatibility result, not a
communication-performance advantage. The next scientific baseline work must
use training-only rate choices and complete mainval outcomes through a common
received-image interface. F9 currently advances the prelocked geometry task
comparison independently; no partial training/AP benefit claim is available.

### Nominal JPEG rates now have fixed real-byte parameters

Full training-only6080 pair encodes select three global JPEG qualities90/39/17
for nominal10/30/50, with pooled real-byte ratios9.9478/30.0625/49.7385 including
framing. Independent server and complete transferred-record audits agree. This
makes the original JPEG comparison executable without per-image rate or AP
selection. A lexical64-pair calibration subset only approximates main rates;
all3769 actual lengths must still be reported. Candidate wire replay is not
claimed by the record auditors. Full shared source/reception caches and task
AP remain the next concrete baseline step. F9 four paired training arms now
finish; final16 AP endpoints are running, so geometry benefit remains unknown.

Six-source-cache engineering now closes the actual byte-to-received-pixel
bridge:12training pairs across JPEG/JP210/30/50, fresh receiver processes,
all33070680native pixel values exact against independent direct-Pillow decode.
Local complete31artifact/actualwire checks pass, largepixelsverifiedserveronly.
Same fixed code launched22614main pairs once(PID265830); completion and AP
pending. F9 snapshot008 has12final endpoints audited; subsequent live read
has13, remaining evaluations continue. No complete geometry benefit claimed.
Both prepared F9 closure/localverifier001 assumed checkout-based AP files;
real manifests useD: and path-only prelock1e92691 introduces002 while keeping
all statistical/state/terminal checks. No result selection or source change.

F9 nowfullyclosed: all16final endpoints and6156transferred artifacts pass,
13360paired training rows and5952inference/prediction records intact. Prelocked
AWGN10 geometryP−genericG=−0.106898pp/P−shuffleS=−0.364217pp; P−U+0.881060pp
cannot identifygeometrybenefit. Preserve negativeprimary; AWGN6positive is
secondary only. Gainchanges aredescriptivelysmall, notacausalexplanation.
Nextrepresentation/policy needsgeometry-specific evidence beyond generichead
or continuedtraining. Main/multiseeds remain. Bothnativeauthorreceivers now
pass24-source-cache engineering frames/25predictions/75artifactlocal closure
with exactshared inputs, noAP; fullmaincachePID265830 remainsinexecution.

All22614 main JPEG/JP2 source pairs now have actual framed wires. Five complete
received conditions and sixth ongoing do not yet establish the full-cache
terminal gate; native all-pixel audit and complete local metadata verification
remain necessary before main inference. A separate complete45228-text verifier
now covers cache lineage, cross-detector input identity, official-metric
records and terminal artifacts; it has not run on main outputs.

The official SRCNN luminance core is now independently translated and checked
for all eight author models on local/shengCPU. This resolves mathematical
filter interpretation and readonly author-weight identity, not the reference
paper's missing10/30/50 compression adaptation or KITTI domain/rate training.
No arbitrary x3model should be presented as the published stereo compressor.

The declared SRCNN source interface now closes synthetic bytes-to-native-RGB
causality and color/geometry conventions with unchanged author filtering.
Both hosts pass four independent check families, six outputs each, complete
record/source checks. Cross-host source bytes match in these fixed examples,
while floatoutput hashes differ. No KITTI adaptation or valid three-rate
performance result follows from the default demo x3model. Fullfloat received
output is retained, including small overshoots, so future quality/conversion
policy must be explicit rather than silently using the RGB8 JPEG cache schema.

SRCNN fixed-budget fine-tuning now passes native eighteen-step engineering
with all actual PNG/patch reconstructions and final optimizer counts. First
checkpoint-output failure retained and repaired without input-barrier exception
or a learning-rule change. All engineering weights discarded; different-batch
losses cannot establish convergence or select a protocol. Full37120updates/rate
still required after current main detector work releases the GPU.

Full public JPEG/JP2 pixel audit passes63,028,337,256 values. Nominal JPEG
10/30/50 compresses11.0188/33.2050/54.6659 on mainval with training-selected
q90/39/17; JP2actual10.0042/30.0180/50.0367. Split-dependent real JPEG rates
are material and must accompany AP; do not change q after this observation or
call the source-only conditions exact resource matches. Actual terminal/local
record closure precedes all main detector forwards; current driver enforces it.

Complete public source-cache closure now authorizes real twelve-endpoint native
detection, rather than engineering or idealized codec metadata. Actual main
JPEG rates remain important companions to eventual AP. All predictions and
native/local verification must finish before treating this as a full result.

The trained SRCNN float receiver closes a distinct datatype boundary: Y/model
float32, color inverse float64, HWC task-cache float32, BCHW input unchanged.
Six synthetic views per host validate trained weight/rate identity, pixel
geometry, CRC, unclipped overshoot and cache tensor causality. Host output
hashes differ, so only independent numerical agreement within each host and
wire/weight identity are established. Full KITTI training/reception is still
necessary.

Full public ROI generation now has an independently audited four-view gate,
with empty ROI retained as undefined and overlap counted once. It is a fixed
sender-side source-quality definition, not an AP-selected mask or an input to
current JPEG/JP2 transmission. Separate declared quality views preserve the
existing unclipped detector input and make original-style quantization explicit.
Quality itself and geometry-specific new-method advantage remain unmeasured.

Full public ROI now closes every7538view identity and union mask, including
670empty views. 9.66193% pooled image coverage is a model-defined region, not
GT object recall. It stays fixed across source conditions and does not enter
current JPEG/JP2 coding or native detector inputs.

Explicit quality formulas now close independent literal reflect-window fixtures
and24actual engineering views. Native integerRGB8MSE independently agrees;
fullSSIM uses the same CPU-verified frozen operator in fresh native replay.
Local all-row source/cache/mask lineage and pooled aggregates pass after a
retained local path correction, without changing measurement or selecting a
codec. Main45228-view quality computation is dispatched but not yet closed.
First fullJPEG10/Stereo-RCNN public endpoint is audited; all12 endpoints remain
necessary before complete comparative conclusions.

Post-F9 primary-literature/boundary audit narrows the following hypothesis: Wasserstein depth-distribution matching is established stereo prior art, not a new communication contribution. Current LIGA samples multi-channel cost features into voxels, and a transmitted probability alone cannot be assumed to preserve its interface. Any next representation must establish usable task information and matched adaptation/resources, rather than add a generic posterior loss to the existing feature codec. No next-method positive evidence exists.

The next SRCNN execution boundary now closes actual source bytes separately
from future GPU reconstruction. Both-host five-family synthetic contract checks
pass, while all six float output hashes differ across hosts. Six actual
engineering wire files pass fresh native PNG replay and complete local byte
verification, including CRC/header/geometry and real compression ratios.
This still cannot establish a usable SRCNN detector input or a main comparator:
engineering weights are discarded, no GPU cache exists, and formal111360-update
training plus complete received-pixel/task evaluation remain necessary.
The live-main guard is tested before CUDA initialization; resource deferral
does not constitute a failed reconstruction or evidence about method quality.
Q0 distinguishes geometry source approximation from channel robustness. A
charged fixed anchor permits received-only, bounded inversion at exactly two
complex symbols/energy2, but three equal-mass quantile particles erase modal
probability. The synthetic48m-separated50/50 case has8m intrinsic W1 even
without noise; uniform support has4.8m. Both-host raw records/physical arrays
and independent scalar replay verify. This does not establish native teacher
sufficiency or task utility. Next test mode-mass flexibility and quantile
valley stability before full task training; neither sphere coding nor generic
Wasserstein approximation is the novel contribution.

Artemis now executes the declared native cost-volume/BEV/point-membership
fixtures on actual RTX PRO6000 capability12.0, terminal0:0 and all transferred
arrays independently replay. The previous pending/build-only boundary is
closed; full detector/MMCV/spconv/task forward-backward remains unverified.

Complete bounded-memory SRCNN acceptance is now prepared and CPU parser fixtures
pass all16,535,340synthetic values and33corruption cases. This resolves local
storage engineering without changing full coverage; actual remote NPZ and
main111360-training/11307-pair acceptance remain unexecuted. Full sourcequality
measurement finishes45228views and all six JPEG detector endpoints are audited,
but complete quality/native/local closure and six JP2 endpoints remain.

Artemis framework integration progresses beyond standalone native operators:
complete frozenPython sources and48official archives are available, and
unchangedMMCV is actually compiling in isolatedCPUoverlay004. The CDN403
occurs on both node types and responds to the download client identifier;
this is operational evidence, not a framework or scientific failure. Full
MMCV/mmdet/LIGA imports, detector forward/backward and sparse teacher GPU
compatibility remain separate unknowns. Preserve the original runtime and
ongoing build inputs rather than silently substitute legacy operators.


Q1's complete both-host23,296-cell proof separates source-law benefit from
wireless utility: explicit modal mass reduces synthetic two-mode intrinsic
W1 from8m to.2m, but minority5% AWGN10 error increases6.6028→9.7127m and
three-mode capacity is lost. Density-gap median stability is a real Q0
limitation; hard-fitting gradients and general task sufficiency remain unknown.
Do not extrapolate a finite-width three-mode perturbation into an infinitesimal
split discontinuity not observed in the predeclared small-epsilon tests.

The native cached-posterior experiment now closes99,840 attempted grid cells,
98,199 valid laws and4,713,552 candidate fits, with every actual output verified
natively and locally by independent CDF/direct-distance arithmetic. Two-node
source approximation improves4.00037→3.23188m, but AWGN10 degrades
12.05784→12.99819m; mean coordinates further degrade to13.80381m, consistently
across four views. This rules out promoting these hard anchor carriers solely
from source fit. It does not refute geometry representation generally, because
the source is an uncalibrated F6b cosine posterior and no task decoder/GT depth
was evaluated. The decisive next question is information usable by native task
features under matched generic adaptation and communication budgets.
An empty-cluster candidate-cost implementation bug was independently caught,
retained and repaired under a separate prelock. Every chosen law and physical
array is identical after repair; only candidate objectives change.

Artemis's unchanged MMCVsm120 compilation and all real framework registry
imports now close actual CPU terminal0:0, with426 frozen inputs and original
runtime unchanged. The missing author-generated version file was filled using
its exact setup rule, reusing the successful wheel. Author484-state checkpoint
93,383,301bytes is fully SHA-verified at the same identity as sheng. Actual full
GPU detector forward/backward and sparse teacher compatibility remain untested;
CPU imports alone cannot establish them.


- Full native RTX PRO6000 author detector and actual supervised3D-loss input
  autograd are executable. Both images have nonzero finite gradients with
  all484authorstates unchanged; no optimizer, teacher GPU or new method claim.
  Evidence: experiments/artemis-detector-framework/analysis-002.md.
- All45228 original-source quality records now close native actualterminal and
  full local lineage/pooled metric proofs. Full pooledPSNR JPEG29.915/27.993/
  26.803dB; JPEG200036.093/30.438/28.652dB. Actualrates unequal; source-only
  quality cannot establish taskAP or wireless benefit.
- Original strict FP64depthmean acceptance remains failed, despite exact
  transfer/state/input/box identity coverage. An explicit nativeFP32 replay
  tests the arithmetic hypothesis without changing originaltolerance.


Complete original-source AP now closes all12 endpoints/45228 frame records,
197185 boxes and45303 artifacts, native and local. At nominalCR10/30,
JPEG2000 has higher RGB PSNR yet lower Car3D R40 Moderate AP for both detectors.
This is measured source-quality/task-ranking disagreement, not an isolated
stereo-depth mechanism; actual byte rates differ and no wireless channel runs.
Main dependency001 omitted24 metric files in its count. Isolated002 repairs
only that assertion/mapping, preserves failed001, reruns no detector/closure,
and launches unchanged SRCNNformal once. PID282374 actually runningrate10.

Conditional feature-measure transmission now supports native task optimization.
GPU00114 cases closes all74988427 saved values. Normalized RBF interpolation
cancels node locations when conditional features coincide; exp/K retains
geometry in the kernel test. GPU002's historical no-cut tolerance fails before
any corrected-codec case; retain criterion/failure. Another coordinate issue
is resolved: native voxel sampler consumes query verticeslinspace2..59.6,
not raw cost centers2.4..59.2. Low-head probabilities remain uncalibrated
selection guides, not true-depth distributions.

GPU003 actuallyCOMPLETED0:0, fourarms32 Adam updates with real supervised3D
loss under identity/AWGN/Rayleigh. All172007982 saved values,107216 gradient
values and32 optimizer checkpoints close natively/locally with independent
Adam arithmetic; author484 frozen. Receive projection uses known spheres,
leaving raw untruncated ZF unchanged. G is only a prior ablation; strongerB
receives the same native prior and has1751 functional parameters vs1650.
Do not interpret two-frame engineering losses as geometry benefit. Full-split
prelock004 fixes P-B primary,3712train/3769val,3epochs,seeds17/23/41. Existing
whole29924-file data audit1980128 continues, with once-only current-task launch
pipeline37826. Full new-method AP, original complete wireless/other source
methods, matched rate curves, multi-seed conclusions and manuscript remain.

Whole dataset audit subsequently completes29924 files/12175258741bytes, disjoint3712/3769. Input prelockefadc6e precedes launch. Pipeline37826 finishes once-only dispatch: seed17 job11424939 actualfull-data86updates, no error; then23/41 array11424940 submitted. All133632 full updates and new validation remain pending.

Fresh immutablefullsnapshot004-001: seed17 job11424939 has535 full-dataupdates, seed23 actualSLURM_JOB_ID11424941 (queue11424940_23) has311, bothactualRTX PRO6000Blackwell/no traceback; seed41 array11424940_41 PENDING. Future terminal audit must use rawjob identity plusarray identity, not equate their textual schedulerIDs.


Bounded full-record audit006 is executable on native and local NumPy runtimes.
Explicit256-step G prefix closes all1421582597 saved values,844800 gradient
values,7587840 complex uses with identical whole row ledger/JSONL hashes.
This does not close44544/seed training or independently certify gradients,
processed RGB or full decoded fields. Actual native005/006 CPU audits and both
exports exit0:0. Original local005 fails exact SNR regeneration: NumPy2.4
uniform differs214/11136 by1ULP. Explicit independent6+12*random matches
native1.26 exactly for every step; isolated006 preserves all exact/numerical
limits and retains failed005. Real first-step acceptance plus consistent
parameter/descriptor, Adam-moment, RX/baseband corruptions rejects all3.

Fresh snapshot004-003: G epochs1/2 complete for seeds17/23;8133/7934 updates
respectively, both actual RTX PRO6000/no traceback. Seed41 remains queued.
Four finite safe epoch checkpoints retained with exact native hashes. Other
arms still pending; no full-training/AP gain. Final3769 inference and official
sheng AP code005 are prelocked/prepared, not executed: original author
Artemis runtime lacks Numba, so preserve it and evaluate fixed text on sheng.
Primary AWGN10 P-B and finalepoch3/publicnoise2027100000+frame remain fixed.


2026-10-05 本轮范围更新与证据：用户要求先不用多seeds，因此当前只保留seed17
RTX PRO6000作业11424939。seed23原始作业11424941及seed41数组11424940_41按
用户指令取消，保留全部已生成记录，不能称为完整结果或算法失败。实际范围快照007
记录seed17共18078/44544更新，G3轮完成，P第2轮；SRCNN压缩率10实际27300/37120，
sheng已认证连接。当前必要训练门槛是44544而不是历史三种子133632。

全3769验证帧的原始PNG哈希/尺寸已与此前完整关闭的六个源缓存逐项对应（22614对）；
builder006对既有状态词的误判保留，007只修正状态词，未改变数量或哈希条件。
纯原作者KITTI writer在真实训练帧7框上原生/本地文本逐字节一致，原生作业11424987
实际正常退出；完整3769收发/文本流式核验006已准备但正式端点尚未执行。

固定首256 G训练记录的全部1198080个显式位置符号完成信道诊断，各预定SNR分箱
全部报告。AWGN从6–8dB到16–18dB的坐标RMSE约5.90降到1.82米；Rayleigh约
10.68降到4.33米。该量是通信包坐标偏移，不是真实物体深度误差、AP、校准后验或
方法优势。高SNR小噪声近似说明位置精度有能量代价；Rayleigh线性化平均的发散
不能等同于有界解码器实际误差发散。P-B还同时存在归一化组差异，先验机制需看
P-G/P-S及正式任务结果。详见position-channel-analysis-006.md和完整诊断记录。

audit007在既有严格状态/数组/Adam/物理标准之上新增各臂逐步实际噪声和衰落
相等检查。固定256组G/P原生探针通过全部151756800个选定物理数组值；首轮
部署路径错误导致CPU作业11425012未启动脚本，失败日志保留，修正部署后代码与
标准不变。此探针不关闭P梯度、完整训练或新方法AP。正式训练及原文无线/匹配
资源比较、其他基线与论文仍未完成；多种子稳定性结论当前不作要求或声明。

新增配对核验本地完整流通过，512记录/151756800个物理数组值，原生和本地逐记录账本哈希完全一致。原生11425015和导出11425016实际COMPLETED0:0；首轮路径失败11425012实际FAILED2:0完整保留。sheng所缺完整数据manifest已部署，SHA与冻结原件完全一致。


2026-10-05本轮实际推进：ECSIC官方模型在独立RTX PRO6000上执行两次原始KITTI
RGB率失真Adam更新，作业11425032、原生核验11425041、全量导出11425042全部实际
COMPLETED0:0。服务器/本地复核全部144085094个保存值及126562968个梯度值，
独立重算RGB/RD损失和每个Adam状态转换；225状态、216独立参数、31640742参数值，
峰值reserve约5.87GiB。原公共权重/官方代码/既有环境未改；独立einops复制及禁用
wandb导入模块只服务新基线，不调用远程日志。两步不代表KITTI正式训练或实际码率/AP。

随后单独预锁固定六lambda(.001/.003/.01/.03/.1/.3)、一个seed17、每候选10轮3712帧
37120更新、共222720更新的KITTI适配，数组11425047并发1。首候选rawjob11425048
实际RTX PRO6000完成5824更新、进入第2轮，无traceback；其余候选排队。仅最终
第10轮合格权重进入固定64训练帧实际rANS字节校准，绝不按验证AP调lambda。
这是明确披露的公开Cityscapes初始化/KITTI适配，非未知曹论文原ECSIC权重。
正式日志和全10轮权重验收不称全部原始梯度/逐步Adam独立重算。
准备auditor002误把225状态当独立参数，尚未执行；保留原版，单独003修正为216并
追加9个真实共享别名严格相等要求。当前训练本身计数正确，不重启/修改训练。

新方法seed17共23243/44544更新，G和P各3轮完成，进入S第1轮；P全部3个权重
已安全读取、全状态有限、优化器步数及原生SHA校验，本地保留。P3 SHA
f52ba24e80525c67f415f05f5dd24339ebef4b1f58b7f14610ea0b6c3e243d1c。
sheng SRCNN压缩率10实际35300/37120更新，原PID282374继续运行。

另修正未来网格完整性：旧005只允许偶数AWGN点，原文要求每整数6–18。孤立008
推理及wire核验保留所有旧源/条件/数值阈值，补奇数AWGN；seed17×4arms×21条件
共84完整3769端点。训练入场明确要求007全部11136/33408真实配对噪声核验及原生/
本地同完整账本。主AWGN10 P-B和固定第3轮不改，尚无新方法验证/AP。所有原文
基线/数字无线/实际资源匹配、机制分析和论文仍未完成；当前不要求多种子。

2026-10-05本轮：sheng认证后持续可连。SRCNN cr10实际训练37120更新已正常退出0，
原控制器282374执行固定完整CPU审计，cr30/cr50尚未启动，三档未完成。真实子进程
282840仍在耗CPU/读盘、内存充足，不重启或误判为完成。快照009-001记录新方法
26982/44544（S第2轮），ECSIC首候选22912/37120（第7轮），两项GPU仍在运行。

原文完整分割输入补齐：原3340+372两个独立YOLO传感缓存的原始每行字节全部保留，
按公开train3712排序，验证集零加入。sheng实际重新读取全部7424训练PNG、
6219939731字节，源SHA/RGB尺寸及31,709框、3,445,709,672个掩码值全部一致；
两个命令退出0，本地完整记录/框/掩码/原生源元数据复核通过。记录SHA
8f4a5ba2b0a35d6da58c580743c88bc7399149c811ed4d8d8da0344d7b668765。
本地没有重读不可用PNG，不声称 fresh YOLO推断、模型训练或AP。

预锁新的原文30档位全3712训练003，保留旧数值架构/五阶段各组件学习率和损失，
先准备前四个语义阶段38轮141056更新，阶段5的167040更新及10/50结构另需执行。
新保存使用safe weights_only读取；原冻结树与历史3340训练不改。CPU检查32份
冻结输入全部通过。仅提交3步丢弃GPU入口验收11425104，因QOSGrpGRES排队；
既有新方法和ECSIC正式作业继续，无新GPU验收通过/正式新训练/AP声明。完整工程
原生及本地审计、实际job+.0退出0后才能启动正式stage1。所有原文基线/无线/
实际资源比较、完整新方法验收/AP/机制及论文仍未完成，目标保持active。

2026-10-05本轮实际推进：首个ECSIC KITTI候选lambda.001完成37120更新/10全轮，
rawjob11425048及.0实际COMPLETED0:0。原生审计初次11425117因未部署已提交的
alias-repair协议，在finally写报告时FAILED1:0，日志完整保留、未生成通过报告。
只部署完全相同协议后，未改训练/阈值/003审计源，11425124及.0实际退出0，
全部37120有序源/RD/有限梯度记录和10个完整安全权重/Adam检查点通过。完整有界
流本地重做955923300个检查点保存值，全部记录及RNG/检查点账本与原生一致；
导出11425126及.0实际COMPLETED0:0，本地pipefail整条进程退出0，完整权重集合未
留在本地。本地证明已部署Artemis。第二候选lambda.003 raw11425119正在训练，
观察7040更新；其余四候选待依次运行。不得把一个候选闭合视为222720总更新闭合，
也不称正式日志独立逐步CUDA梯度/Adam数值重放。

另外真实完成P6EK模型绑定码流格式004，本地/sheng各两份实际旧源码流的熵流体
逐字节保持一致，新格式实际166字节头/CRC，65个跨模型/恶意CRC正确头/截断等
失败全部拒绝，两主机fixture账本相同。旧Cityscapes P6EC固定MODEL_SHA及其全部
历史源/接收证据不改。预锁005最终六模型登记及新byte-only神经图，安全读225状态
及9别名、216唯一参数，准备E0/HE0/HD1/D1全18数组和原尺寸float裁剪。实际未完成
候选注册请求被拒绝，未创建任何权重输出；这不是适配模型神经解码/率校准/AP证明。
后续仍须全部六最终候选验收、具体registry身份、数值神经工程门槛，再固定64训练
帧真实rANS字节选择原文10/30/50档位。

最新不可变快照010-001：新方法seed17共31697/44544，S第3轮，原文新完整分割GPU
3步入口11425104仍因QOSGrpGRES排队；两正式GPU作业保留。sheng SRCNN10训练完成
后的真实CPU审计子进程282840继续运行/耗CPU；未重启或假定三档完成。完整原文
无线、资源匹配及所有新方法AP/机制和论文仍待完成，目标保持active。

2026-10-05机制推进：明确坐标相位的可靠性取决于发送坐标和信道，而不只取决于
平均SNR。009推导AWGN及原perfectCSI/iidZF Rayleigh的精确相位密度，再对冻结
decoder的环绕与截断积分；010仅修复NumPy scalar JSON序列化，原失败保留。
本地/sheng各26物理条件×5固定坐标全部130数值检查通过，每主机136314880坐标
观测；最大MC误差2.650SE，数值跨版本一致。10dB中点RMSE为4.219m/8.060m；
边界为2.985m/9.022m，两信道boundary clip概率都0.5却误差不同，说明clip频率
不能单独当可靠性指标。Rayleigh正端点的对端跳转尾事件解释约38.60/81.40m²
误差（事后解析解释，不新预锁端点）。不称物体误差、实际theta平均、AP收益、
校准不确定性、Shannon RD或新的Gaussianratio数学定理。下一项科学判断仍需
固定finalepoch3全3769 AP及同detector/完整无线/实际资源控制，不改primary。
新方法真实34774/44544，G/P/S各三轮完成，B首轮；S1/2/3全部小权重安全哈希/
finite/Adam步数核对通过，不等于全007训练证明。ECSIC候选1真实21184/37120；
SRCNN cr10完整CPU审计仍实际运行。原文GPU三步入口仍配额排队，不重启训练。

2026-10-05完整记录进一步闭合：ECSIC前两候选均完成正常训练和所有10epoch
安全状态的native/local全查，共74240记录/20检查点，各955923300保存值；训练/
native/export/.0和localpipe均正常0。余4候选和最终bytecalib/AP仍不可省略。
原生CPU全44544训练proof007已依赖主train11424939排队11425240。012全84
执行路由严格防止partialproof入场，两CPU实际提前拒绝，未产生任何新AP。
冻结方法说明明确S是guide空间roll，RBF权重exp/K不跨节点归一，B/Psphere
组不同。固定finalepoch3的S宽34.667m相对P7.474m更宽，但参数变化只能说明
学习到不同basis，不能替代fullvalidation或建立因果优势。PragComm2024v1中
已存在taskfeature/dictionary/directperception和明示的noiseless abstraction；
2026版出版身份已核实，不声称其全部正文等同v1。当前与geometry/noisychannel
的设置差别依然不是充分创新证明，原文baseline/radio/actualcompute控制仍必要。


2026-10-05执行衔接：sheng用户认证后SSH已确认可达，原SRCNN cr10完整CPU审计
仍是实际进程282840，CPU累计从1:14:35增到1:20:53，未重启、未当作通过。
新方法最新实际快照013-002为42789/44544，B第3轮，其余G/P/S各3轮完成；
B1/B2小安全检查点保存值各5337全部finite，SHA与原生相同、Adam3712/7424。
完整原生007审计11425240之外，新增预锁013全44544本地有界流，实际导出任务
11425814依赖11425240，前台本地session43608存活等待；原生和本地全量都未通过。
ECSIC第三候选lambda.01真实20416/37120，raw11425607；新增预锁010本地全量
37120/all10流式复核11425840依赖native11425708，前台session9438等待。
原文完整分割GPU3步入口11425104仍配额排队，其预锁004CPU审计11425880
依赖该GPU实际正常结束，原生/本地工程门槛及fresh正式stage1还未成立。
三项新增任务只衔接既定完整范围；没有新的检测AP、模型校准、方法优势或项目
完成结论。四组固定epoch3/全3769主比较和完整原文radio/actualresource仍必须执行。

2026-10-05完整证据里程碑014：新方法四组各3轮已实际完成44544更新，GPU11424939
/.0正常0，report SHA5a0f3cb6、updates SHA4899619f。全部12小checkpoint安全读、
nativeSHA/finite/Adam3712/7424/11136检查通过。全007 native11425240仍运行，最新
22336条；local43608/export11425814仍等待，不提前生成012门槛，不称AP优势。
原文public30三步GPU11425104及native11425880全部正常，完整原生/本地003工程
状态/Adam/counter一致，正式fresh阶段1已由006门槛准入11426592，最新14208/44544，
全阶段audit11426847已排队。第一次sbatch因已完成短作业超出Slurm controller
MinJobAge拒绝；原失败保留，只对同一wrapper去掉失效调度依赖，wrapper继续检查
实际sacct和全部proof。不存在训练重启、工程权重初始化或降低审计条件。

ECSIC第三候选lambda.01的37120/all10全部native/local955923300保存值核验闭合，
train11425607/native11425708/export11425840及.0正常0，local9438实际退出0，全部
metadata逐项一致。前三候选累计111360更新/30checkpoint；第四lambda.03实际
11426641观察24256，native11426848及export11428494/local47174等待，另外2候选
未完成。六最终模型注册、同数值环境真实神经编解码、固定64训练帧字节校准及
全验证AP仍必要。均不是独立逐步CUDA梯度重放证明。

实际source架构005三档两端CPU原尺寸合成RGB/编码器50梯度finite通过，30全部768
初始化状态和同host实际输出与冻结控制完全相同。跨host三个输出hash都不同，
probe未封存原始合成输入hash，不能归因于某个运行时或声称跨环境像素bit相同。
10/50真实source训练、stage5/mainwire仍未完成。SRCNN cr10服务器端全37120更新
1187840独立patch/7424源views/148480读取和finalAdam通过，SHA32413edf；controller
282374继续cr30，最新log9900步，不把native通过视为local/全cache/AP闭合。
核心开放问题保持：几何语义表示是否在实际同资源/同detector条件带来检测收益，
机制是否能被配对消融支持；现有训练完成与理论坐标误差均不能替代这些证据。

2026-10-05源loss008：实际发现冻结stage2的ROI padding固定6不适配50的factor8，
隔离新source_loss采用lcm(g,k)填充，原30源码/当前训练完全不改。预锁后本地
27933及Artemis11429177/.0均正常0，三档×六固定source phases完整grad检查通过，
合成263×265而非真实KITTI。两个raw input SHA相同；每host30的全部loss/outputs
/fullgrad ledger逐bit与冻结控制相同，18conditions模型state不变/全activegrad有限。
10/50分别16个keygrad含1908/11808values，表明确实不同空间架构。跨host输出和
gradient不假定bit相同。正式10/50训练入口/auditor008已隔离准备，需各3真实GPU
工程更新、正常terminal及完整native/local proof后freshformal；stage5/radio待实现。

2026-10-05第四ECSIC候选lambda.03闭合012：训练11426641/.0正常37120/10，native
11426848及export11428494/.0均实际正常0；local47174完整pipe退出0，native/local
全部metadata及37120/10/955923300保存值/完整账本一致。前三的结论不变，现在
四候选累计148480记录/40checkpoint，仍不是全六或实际bytes/AP。剩余4/5统一
预锁013保持原003wholecriteria，actualnative array11429217 aftercorr11425047，
export4/5真实11429221/11429222，local55964/69803为前台有界等待，不假设通过。

2026-10-05里程碑017：main全007 native11425240/.0实际正常0，44544/11136/33408
完整保存值及物理配对核验通过；本地43608开始处理并观察15808条，尚未完整退出。
因此完整训练门槛014、validation012及任何检测优势仍未成立。原文30 stage1观察
39360，ECSIC第五候选lambda.1实际raw11429214观察34432；都不是终态。

接收端015有序有限网格后验、016完整逐试验保留、017独立fsum核验在local/sheng
均实际正常0；每host20条件1310720先验物理试验/14417920保存误差值完整复核。
同先验下10dB ordered坐标MSE为AWGN12.93095/Rayleigh26.53775m²，hard为
16.12337/69.35909m²。这里的m²是57.6m归一化潜坐标误差，不是KITTI物体深度。
独立uniform-slot控制与joint-ordered的边缘先验不同，不能将全部差距归为节点联合
约束；下一项机制控制必须匹配真实slot marginals。后验基函数期望优于mean-only
也只在固定合成系数/先验成立。实际训练节点先验、feature相关性、额外算力、完整
3D AP与通信资源仍未验证；新原型没有改变既定84主验证端点。

2026-10-05里程碑018：原文30第一阶段实际全部44544/all12已native/local完整通过，
实际train11426592/native11426847/export11429371均正常0，local16291退出0；第二
阶段11429413已实际960/37120运行，原003架构/loss/schedule和freshAdam未改。
第五ECSIC candidate4/.1全部37120/all10保存值955923300及完整metadata两端相同，
实际三作业与local55964均正常0，现在五候选185600/50闭合，最后一候选15872。
这些是基线完成进展，不能替代真实rANS字节率校准、物理链路或检测AP。

用户允许有部分实测结果时先写LaTeX，故提前形成ICC2027工作稿并内置编译成功，
没有等到科研结论完成。主线忠于实际conditional-field/received-only实现；已有
source-only的PSNR/AP排名差异和synthetic prior机制写入，四主比较AP保持pending。
稿件保留固定预算、expensive front-end、global sigma、不校准guide、feature/node
相关性忽略、idealCSI、matched-marginal控制尚缺及receivercost未测等限制。引用
基于Crossref/DataCite/CVF原记录；不声称当前已经是投稿终稿或最新文献全覆盖。
