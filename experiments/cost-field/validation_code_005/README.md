# Fixed final validation executables

`infer_GPU.py`: Artemis RTX PRO6000, unchanged004 codec/author runtime,
final epoch3, complete3769 IDs, received-only cost+appearance, complete wire
records and original KITTI writer. Requires complete native and local training
proofs, each44544 updates/seed and actual terminal0:0. Does not read labels or
compute AP. No detector/codec optimizer is constructed.

`evaluate_sheng.py`: original official Numba/CUDA KITTI evaluator on sheng,
complete fixed3769 native text files and source/GT hashes, after actual complete
inference terminal. Produces primary Car IoU0.7 R40 plus secondary0.5 R11/R40.
Its output remains pending actual AP process exit and full wire/local closure.
No custom replacement of rotate-IoU or AP implementation.

These files are prepared and syntax checked; no full new-method validation/AP
has run. Do not substitute the256-step engineering proof for full proof. Start
with AWGN10 for all G/P/S/B and17/23/41; use every frozen final checkpoint and
then identity+complete radio matrix. No AP-based checkpoint choice.

Training audits: `data/provenance/audit-cost-field-full-006.py --mode native`
and `--mode stream` over `--mode export` ordered tar data. Default count44544;
explicit smaller count produces engineering-only proof. All raw arrays stay
on Artemis; complete local stream checks each saved value without retaining
the large dataset. Use unique output paths and preserve failed reports/logs.

Protocol: `../final-validation-protocol-005.md`, unchanged primary contrast
P-B Car3D Moderate R40 at AWGN10. Transmission19 complex symbols/energy19 per
pooling site; genericB has1751 parameters versus geometry1650. Report sender
cost-volume compute, lack of detector portability, incomplete matched-rate and
original radio comparisons. Prepared scripts establish no performance gain.
