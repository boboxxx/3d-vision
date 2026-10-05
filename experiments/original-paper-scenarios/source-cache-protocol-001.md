# Six complete original JPEG/JP2 source and reception caches

Prelock before new compressed main-validation observations. This addendum
executes the source/reception portion of the six fixed source-only conditions
in matrix-001.json: JPEG and JPEG2000 at nominal10/30/50. All3769 public main
validation IDs must appear in each condition, in the fixed split order. Clean
main-validation detector results have been inspected previously and are
disclosed; these compressed outputs are new. No detector/AP/GT/ROI/SSIM/model
reads or hyperparameter selection take place in this stage.

Use /mnt/d/paper6/data/kitti/ImageSets/val.txt, SHA256
657ac4bcc1e156e5b106a4ca18e1f88e012787ea1d2b5d0adeea97fee903fa86.
The training engineering gate uses exactly000000/000003 (in the sealed3340
internal training fold), with the same six conditions and code. Never use
engineering outputs as main outputs. Preserve any failures under their unique
ID; a changed implementation needs a prelocked narrow repair and a new ID.

Global JPEG quality10→90,30→39,50→17 is sealed by completed calibration:
manifest ad41d390c7084a7279920cce7d881f327071630b3bf8852c65dbc88033379d62;
independent native audit e014bc8b45d172189eaaa43b7aeca7abb1cbdd1a34401fe59aa5ecca28dc5ecd;
local full-record verification320570301851fbabd5024ee70253839c7bcdcad517b06e06248a326f773dcceb.
JPEG4:2:0, optimize=false, progressive=false. JP2 boxed, irreversible=true,
MCT=1, single quality_mode=rates layer10/30/50. Reuse unchanged old
image_codecs.py/contracts.py and their native seven-check predecessor.
No per-frame rate, image-quality or detection-quality search.

Pin sheng Python3.10.15,NumPy1.26.3,Pillow10.2.0,libjpeg6.2,
libjpeg-turbo3.0.1,OpenJPEG2.5.0. No dependency or eight-tree F9 source changes.
CPU-only, four encoding workers at low process priority; up to four independent
audit workers. All large wires/caches live on D:, with100GiB physical free
space required before main launch. Sources and runtime must remain fixed.

First source encoding seals all six sets of complete real P6SBv1 wires. Record
each native PNG hash and full image-array identity, both codestream lengths,
20-byte framing/CRC, total bytes/raw_RGB8_bits and exact wire SHA. Source-only
has no modeled PHY/noise/digital code/pilot/ROI side link: complex channel uses
and PHY energy are undefined, not an invented favorable zero-rate comparison.
Actual source bytes, including framing, remain fully reported. All pairs use
full native RGB8 geometry and unchanged codec options. Encoding failures stop
the experiment and remain visible; do not skip or replace frames.

Then six fresh receiver processes, one per complete condition, consume only
that condition's complete wire files, public split/frame identifiers and fixed
decoder configuration. Deny all PNG,GT,calibration,model/infos/foreignNPZ reads.
Wire JPEG/JP2 headers alone supply native dimensions; no clean-image shape,
source metadata probabilities or source-file fallback may enter reception.
Freshly decode each complete framed wire, preserve HWCuint8 left/right exactly,
and write one uncompressed NPZ per pair plus received-array/file hashes.
No resize, augmentation, denoising or learned-output clipping. The eventual
detector conversion is float32 /255 with existing native detector preprocessing,
and requires a separate inference addendum. Both downstream detectors will
consume the same fixed received cache, avoiding platform-dependent JP2 pixels.

Independent server audit must parse framing/length/CRC itself, decode both
streams with fixed native Pillow directly, compare every cached pixel exactly,
freshly hash all source PNGs/wires/caches, and verify all six complete ID sets,
byte accounting, runtime/source identities and receiver input barriers. It may
read clean PNGs solely as an external source audit, never as receiver input.
Large native pixel checks stay on sheng. Terminal/source/artifact closure and
independent complete transferred-record checks close the stage; they do not
constitute local raw pixel replay or six completed AP conditions.

Start full main3769 only after the two-training-pair engineering source/receiver/
independent audit and transferred metadata checks close on the final code.
No restarts or source changes during a running unique execution. This stage
prepares the required full baseline inputs; complete detector AP, source-quality
metrics, digital SNR/fading outcomes, ECSIC KITTI adaptation and novel-method
main/multiseed comparisons remain required work.
