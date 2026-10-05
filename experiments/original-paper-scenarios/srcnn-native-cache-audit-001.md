# Native SRCNN source/cache execution and audit

Additive execution clarification before native SRCNN cache or detection results.
The frozen float-reception protocol0626389a87f72979920dd08f8d7df173bf809ed87e5f3601ba448bb90fa5625d,
source bytes, training002 and receiver implementation remain unchanged.
Existing synthetic float checks and discarded engineering weights were
inspected; no main SRCNN reconstruction or AP has been observed.

An isolated five-file native-cache package implements encoding, rate-specific
received-only reconstruction, fresh native audit and CPU execution-contract
checks. Encoding records every source PNG SHA/RGB identity, actual32-byte
framing and payload, wire CRC/SHA/size and pooled raw-to-wire ratio. Every
native pair/rate is encoded once. Receiver obtains H/W solely from the wire,
loads the already validated static six-state checkpoint, installs the frozen
read barrier and writes exact unclipped float32 NPZ pairs. Sender metadata
only lists routes/wire identities; clean shapes and arrays are not receiver
inputs. Source-only PHY use and energy stay undefined.

Native independent audit separately spells out literal header/CRC parsing,
RGB8 reduction and mode-F resize/color conversion. It reconstructs each full
native output from six loaded checkpoint arrays by direct Torch padding and
convolution, without calling the production receiver or model forward.
The underlying Torch convolution backend is shared; independent synthetic
SciPy kernel checks remain the earlier CPU gate, not a claim of a second full
SciPy or MATLAB implementation. Bound full output differences by3e-5 absolute
and record the observed maximum. Exact actual received cache/array hashes,
range statistics and six unchanged state hashes are checked for every pair.
Fresh PNG→wire checks and typed cache/tensor identity cover all native pixels.

Engineering uses only000000/000003 training pairs and all three discarded
six-update engineering002 weights. Main uses all3769 ordered public validation
pairs only after full formal training002 actual terminal and all111360 local
record checks. No main encoding before that gate. Both scopes require the
existing float CPU proofs/current source identity. GPU receiver/audit require
the main JPEG/JP2 cycle to have exited, no competing compute PID, initial
free>=12GiB, and peak reserve+2GiB<=initial free. They never interrupt it.
Formal training has priority once its pre-existing release chain dispatches;
engineering reception can execute after that training terminates.

CPU contract fixtures use synthetic images and actual discarded checkpoints.
Verify literal bytes independently, all three rate routes, static model and
unclipped cache identity, source/checkpoint mismatch rejection and explicit
barrier activation. No actual GPU execution, native KITTI reconstruction,
formal weight eligibility, source quality or AP is inferred from these checks.
Every failure retains its run ID; no retries or checkpoint/resize selection.
Native terminal/local all-record closure precedes downstream engineering/main
detector adapters and any result release, which remain separate work.
