# Actual-byte training calibration fixes the three JPEG qualities

The complete prelocked search selects JPEG qualities **90 / 39 / 17** for
nominal raw-RGB8 compression targets **10 / 30 / 50**. The measured pooled
training ratios are **9.9478336 / 30.0624702 / 49.7384857**, including the
20-byte P6SB header and CRC. These are approximate operating points; they are
not exact ratios for every image or measured main-validation ratios.

Protocol fc7ae0d precedes implementation 936ecb7. Native CPU PID 262111 runs
2026-10-04 20:22:01–20:23:19 UTC and is actually absent at independent audit.
All 95 integer qualities use the same fixed 64 training pairs, for **6080
actual pair encodes**. The encoder is the unchanged previously checked Pillow
implementation: native dimensions, JPEG4:2:0, optimize=false, progressive=false,
two real codestreams and complete serialized framing. Python3.10.15,
NumPy1.26.3, Pillow10.2.0, libjpeg6.2/libjpeg-turbo3.0.1 are fixed on sheng.

The lexicographically first64 internal training IDs are disjoint from the372
codec holdout and3769 main-validation split. All128 PNG files are freshly
hashed before and after. An input barrier forbids other KITTI files, model/
infos/NPZ inputs; there are no GT/model/validation/AP/wireless observations.
The sample is a fixed lexical subset, not a random representative KITTI sample.

Each quality uses the pooled ratio total raw bytes / total complete wire bytes.
The selected quality minimizes absolute log ratio error, with lower-quality
tie break, as committed before measurement. Independent server audit checks
all6080 records, all128 native source identities, coverage/order/accounting and
all three selections. Independent local verification repeats complete record
coverage, pooled ratios and selection from the transferred raw JSONL. Candidate
wires were measured by the sealed native encoder but are not all retained or
re-encoded in the auditors; local native PNG replay is not claimed.

Evidence:

- [Native manifest](/Users/chen/Documents/ChatGPT/paper6/data/runs/original-jpeg-rate-calibration-001.json)
- [All6080 byte records](/Users/chen/Documents/ChatGPT/paper6/data/engineering/original-jpeg-rate-calibration-001/calibration.jsonl)
- [Independent native terminal audit](/Users/chen/Documents/ChatGPT/paper6/data/provenance/original-jpeg-rate-calibration-001-audit.json)
- [Independent local verification](/Users/chen/Documents/ChatGPT/paper6/data/provenance/original-jpeg-rate-calibration-001-local-verification-001.json)

These global qualities must be frozen for subsequent compressed main images;
no per-frame rate/AP/quality retuning is allowed. JPEG2000 single-layer rates
10/30/50 are already fixed separately. The next execution addendum should
prepare all six complete3769-frame source/reception caches on the same sheng
decoder, before detector inference. Actual main bytes/energy, source quality,
detector AP and the complete digital channel matrix remain unmeasured here.
