# Native RGB CPU endpoint engineering, fixed fresh initialization

Prelock before this probe: use solely the original formalstage1 initialization
snapshot SHA f4d0d5d5bcc203ee5fd0554a0e691b2a530c56cb55d48fe21b6fc0f97331d3cb,
all768 fixedstates, no optimizer updates. It contains seed17 fresh architecture
plus official SpyNet, not final trained communication weights. CPU is used to
avoid changing concurrent F6b/original GPU workloads; set2 CPU threads, require
at least8GiB host MemAvailable before construction. Do not alter GPU training.

Use only first ordered sensor-ROI holdout frame000036, nativeuncropped RGB,
audited image/ROI/fold identities, no label/GT/calibration inputs for codec.
Identity andAWGN10dB engineering attempts, separate deterministic generator17.
Run the full original semantic and9-channel air receiver, not semanticfixture.
Use adapter's explicit received-observation-only boundary, allmodules eval,
no_grad; broad WirelessVariant.forward and cleansemanticMSE forbidden. Preserve
all768 state hashes before/after both attempts. Trace sender semanticencode,
received payload andsemanticdecode; erasure must preventdecode andretainalluses.
Native official Stereo-RCNN preprocessing can run ondecodedRGB onCPU, without
loading/detecting a 3D model or producing AP. Record decodedfloat extrema,
actualsymbol/control/energy counts, sensor geometry, processedtensor hashes,
readonlystates and original21/sourcehashes. Check cleanRGB relay equivalence
separately against official BGR path. Do not infer trainedtaskperformance.

No LIGA/nativeGPU/3D solver gate is fulfilled by this probe. Formal inference
still requires finalaudited stage5epoch45 weights and both native detector
endpoints under the integration protocol. All probe artifacts are engineering
only; never reuse its initialized or decoded values as trained baseline data.
