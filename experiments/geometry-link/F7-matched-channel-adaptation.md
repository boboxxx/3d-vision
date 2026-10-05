# F7: matched clean continuation versus AWGN task adaptation

Prelock after the fully closed F6b result. This is a two-arm controlled follow-up
to the observed 6.8834587 percentage-point identity/AWGN10 Moderate AP gap.
Hypothesis: explicit noise adaptation improves AWGN10 task performance beyond
an equally exposed clean continuation. This tests channel training, not a new
resource allocation mechanism, general superiority, or novelty.

Both arms start from the sole complete 535-state F6b final checkpoint:
`/mnt/d/paper6/runs/stereo-native-task-seed17-002/checkpoint_epoch_1.pth`, SHA256
`77bc73021a139099e3d0f9c12e0fba8b17eb62f9b9c243eb8f0ad099bd6e8867`.
No earlier failed/engineering/intermediate learned weights may initialize either
arm. All 519 student/detector states remain exact fixed; only the same 16 existing
codec parameter tensors update. Fresh identical AdamW per arm, LR 1e-4, weight
decay 1e-4, clip10, one native epoch, exactly 3,340 actual updates. Native augmented
training split, batch1/four spawn workers, model/data seed17 and all modules eval;
same native classification/box/direction/IoU loss and target barrier as F6b.
Preserve native legitimate empty-GT handling, no skip/dummy targets, no teacher,
depth/2D supervision, reconstruction/imitation loss, or private clean bypass.

Control: identity channel. Treatment: AWGN channel. Both use the identical fixed
per-update SNR sequence U[0,20] dB from a dedicated Python Random(1707), length
3,340. The identity channel ignores SNR physically, but records the same public
SNR condition. Treatment noise comes solely from a dedicated CUDA Generator(1708),
never model/data global RNG; serialize the initial/final noise RNG state and
schedule identity. Equal native data/augmentation order is verified by per-step
frame IDs and hashes of augmented left/right, GT, image_shape, random_T if present,
and public calibration. Hashes are audit outputs and never model inputs. If
paired raw fingerprints differ, preserve results but do not claim matched
exposure; diagnose before any new independently prelocked continuation.

Both retain the exact stereo/app layout: stereo[1,32,320,1248] twice,
app[1,32,80,312], 62,400 complex data uses per frame (49,920 stereo/12,480 app),
joint average energy1, no header/pilot for identity/AWGN. Receiver builds native
cost after communication from received features only; GT appears only at native
3D head. Record chronology and all five cost/feature/symbol gradient branches,
actual losses and nonzero/finite counts, full state and actual Adam moments/steps.

Predetermined endpoints: the final checkpoint only, both identity and AWGN10dB
on the complete ordered 372 internal hold-out frames, seed17, unchanged native
NCCL/DDP inference/AP/prediction/calibration/channel/feature/state audits. Four
evaluations total, no best-checkpoint, SNR/seed selection or main-val tuning.
Primary comparison: AWGN10 Car Moderate3D AP_R40(treatment minus control) in
percentage points. Report both clean APs, all E/M/H, clean/noisy tradeoffs and
all failures. A positive single-seed difference is limited exploratory support,
not statistical significance or final method superiority. A nonpositive result
does not justify silently adding epochs or choosing another training SNR.

Before formal: six native engineering updates per arm from that same sole F6b
parent, independently audited all535/519frozen/16actualAdam6/native gradient and
paired data fingerprints; six steps include the previously identified empty GT.
All engineering weights excluded from formal initialization and all 12 updates
charged as research exposure. Compare these branches before launching formal.
Run arms sequentially to limit GPU contention, each with NVIDIA physical free
>=10GiB and peakreserved+2GiB<=initial physical free. Never interrupt original
PID26642 or other users' jobs. If margin unavailable, defer GPU execution.

The native Stereo-RCNN RGB engineering probe must reach terminal source closure
before changing root executable sources for this implementation. The original
21-file trainer and original formal protocol remain frozen throughout. Commit
implementation, meaningful regression checks and engineering evidence separately
after this protocol; freeze all root executable/upstream sources from formal
launch through both arms and all four evaluations/full artifact closure. If
engineering reveals an interface problem, retain failure evidence and explicitly
prelock the narrow repair before rerun, never tune using hold-out AP.

All old F6/F6b protocols and data remain unchanged. Existing F6 compatibility
defaults must retain strict identity/F5b initialization behavior. New F7 modes
must explicitly select parent, channel and schedule; independent auditors must
reject substituted parents, mismatched per-step SNR/fingerprints, optimizer
states, source/protocol identity and received-only boundary violations.

Limitations: internal hold-out overlaps author detector pretraining; single seed;
uniform fixed resource layout, no uncertainty allocation or learned-risk claim;
original trained wireless baseline and matched rates/exposure still incomplete;
no fading/digital-channel or final main-validation evidence from this experiment.
