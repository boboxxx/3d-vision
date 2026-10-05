# Actual received ECSIC bytes now produce causal stereo RGB caches

The prelocked two-source engineering chain is complete: all 20 fixed physical
attempts retain their original outcomes and resource charges. Eleven received
packets each run a fresh unchanged neural decoder and a fresh crop process;
nine erased stereo pairs produce no neural output or fallback. This establishes
the reception interface, without measuring KITTI detection AP or reliability.

Protocol e3ac2d1 precedes implementation 0be89ec and the unique CPU launcher
13fac9d. Actual controller PID 261130 started at 2026-10-04 19:59:04 UTC and
finished both native commands at 20:00:50 UTC. Fresh terminal closure finds
that PID absent and both commands exited with code zero. No rerun, new channel
draw, source encoding, model update or GPU execution occurred.

All 22 actual accepted P6SB/P6EC files were freshly hashed on Artemis, after
local transfer, and on sheng. Receivers consume the corresponding received
container and its public model/config/CDF/dimensions, with source-image/NPZ
reads denied. The sealed causal zL → zR → yL → yR decoder calls HD/D once and
E/HE zero times. All 225 model states remain exactly unchanged.

The independent server audit compares every one of the 18 reconstructed arrays
for each received packet against its sealed stage-B reference: **198 exact
array comparisons**. It also checks every value in **22 cropped views** against
the corresponding actual reconstruction. KITTI frame 000000 has public native
370×1224 dimensions and padded 384×1248 dimensions; top-left cropping preserves
FP32 values, including values outside [0,1]. There is no resizing, clipping or
RGB8 conversion. Detector-specific conversion is still a separate interface.

Closure freshly hashes all 67 native manifest/output files. It snapshots 45
small reports/logs and transfers those with the other provenance and received
bytes. The independent local verifier checks all 75 listed artifacts plus the
closure, source identities, 20 outcome/resource records, public wire headers,
22 child PID chronologies and 225-state/18-array/crop metadata. Large raw arrays
remain on sheng; local metadata verification is not a second raw neural replay.

Evidence:

- [Actual terminal closure](/Users/chen/Documents/ChatGPT/paper6/data/provenance/ecsic-neural-reception-native-001-closure.json)
- [Independent server raw-array audit](/Users/chen/Documents/ChatGPT/paper6/data/provenance/ecsic-neural-reception-native-001-audit.json)
- [Independent local verification](/Users/chen/Documents/ChatGPT/paper6/data/provenance/ecsic-neural-reception-native-001-local-verification-001.json)
- [Native CPU gate](/Users/chen/Documents/ChatGPT/paper6/data/engineering/ecsic-neural-sheng-CPU-001.json)

The public Cityscapes λ=.01 model is still an engineering checkpoint, not the
original paper's recovered KITTI adaptation or a matched 10/30/50 operating
point. Full 3769-frame shared reception caches, detector AP, training-only
rate selection and the complete original scenario matrix remain required.
