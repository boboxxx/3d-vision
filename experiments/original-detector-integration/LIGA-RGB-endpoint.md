# LIGA received RGB endpoint: native coordinate and module gates

Prelock before new endpoint implementation/engineering or any original-codec
AP observation. Retain the original RGB downstream integration protocol and
all current F6b/original sourcefreezes. This note clarifies the author484-state
clean LIGA endpoint: its ResNet/feature neck are the required image backbone,
not a privileged teacher in this mode. They must execute twice. Only student
feature distillation, privilegedLiDAR teacher and training losses are excluded;
students/geometry links/RGB JSCC links are disabled. Do not remove the original
image backbone to make a teacher-call counter pass. Nativeeval2D prediction
may remain if present in the originalmodulelist; no2D/depthtrainingloss orGT.

Received float32 native RGB is multiplied255 without clipping/quantization,
converted toHWC and passed through exact original eval crop augmentor and
StereoDatasetTemplate.collate_batch. Lock onlyrandom_crop with min/maxx0/0,
y1/1 andmax320x1280; retain native offset(x1,y1), float32 ImageNet normalization
andright/bottom zero padding to32. Clone fresh public calibration percall:
never cumulatively offset the caller's calibration across identity/AWGN/detectors.
No GT, points, LiDAR, roadplane or depthtargets enter thispreprocessing helper.

The native __getitem__ restores original image_shape after prepare_data:
keep originalH,W, not croppedH,W, for finalKITTI projection/clipping. Retain
calib_ori separately for nativepredictionwriter; sender/detector inference
boundary still onlySENSOR_INPUT_KEYS and croppedcalib. Test firstordered
frame000036 exact cleanrelay RGB tensor andcalibration/meta agreement against
the unmodified nativeevaldataset, not just a copiedformulafixture. The reference
dataset may load GT/LiDAR as its normalpreparation, but these values cannot
enter the codec/helper/model inference call. No referenceAP is produced.

First meaningful local tests use exact AST bodies of nativecrop/collate plus
pureCalibration; separate shengCPU engineering uses realnativeimports/full
native dataset reference, cleanrelay, out-of-range decodedfloat preservation,
repeatedcalibrationclone, original image_shape andpadding. Use fixedfirstframe,
not fullAP or selection. These close preprocessing only; full484-state strict
pretrainedload/read-only sensoronlyGPUforward still required. Schedule the GPU
check only in a measured physicalmargin window without stopping or changing
F6b/original native training or their source identities.
