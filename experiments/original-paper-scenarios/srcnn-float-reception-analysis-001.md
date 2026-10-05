# Trained SRCNN byte-to-float interface: CPU gates closed

Protocola612bd7 precedes implementation1aef4c8. Reception uses the existing
real32-byte P6SR frame and corresponding six-state float32 engineering002
checkpoint at each10/30/50 rate. The fixed Y float32 model, float64 color
inverse and one final HWC float32 cast preserve native geometry and overshoot.
Shared task-cache tensors apply no divide-by255, clipping or quantization.
The current low-level loader validates static checkpoint/run metadata; it
does not independently authorize a formal run without full training closure.

Both hosts pass six CPU families with six synthetic32×48RGB views each and
27,648 output values each: strict trained-state metadata/rate rejection;
independent SciPy convolution of all loaded filters; separate header/Pillow/
color inversion; malformed header/length/CRC rejection and received-only
causality; readonly six-state and exact float-cache/tensor identity; guaranteed
overshoot and real clean/GT/foreign-input read barriers. No GPU is used.

| Host | Maximum independent Y error | Maximum independent RGB error |
|---|---:|---:|
| Local |1.37090683e-6|1.60932541e-6|
| sheng |1.13248825e-6|1.31130219e-6|

Errors pass the locked2e-5 model and3e-5 derived-color checks. Complete local
verification checks all twelve transferred view records, three actual trained
checkpoint identities, both source trees, unchanged20 training/core/interface
files and complete engineering closure. Actual wire and weight hashes match
across hosts. All six floating output hashes differ across host runtimes;
no cross-host bitwise output equality is claimed. Each host independently
computes its numerical reference and verifies its own exact cached tensor.

`cpu-local-001` is the retained development report before adding explicit Y
error reporting; its code fingerprint differs and it is never a current gate.
Final local/native reports are both002 and match committed source1aef4c8.

These are actual six-update engineering weights and synthetic input checks,
not formal37120-update training, full native KITTI received images, source
quality, detector AP, digital delivery or new-method evidence. Native full
geometry/cache/detector engineering and formal training gates remain required
after current main JPEG/JP2 detector work releases the GPU. All engineering
weights are discarded.

Evidence: `srcnn-float-reception-cpu-{local,sheng}-002.json` and
`srcnn-float-reception-transferred-verification-001.json`.
