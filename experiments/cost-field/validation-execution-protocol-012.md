# Execution of the complete fixed validation008 grid

Execution wrapper only; all004 training, finalvalidation005/008 and full
proof007 criteria are unchanged. Prepare before any final validation outcome.
This preparation does not claim that training or whole proofs are complete.

Use the immutable84-endpoint validation-plan-008.json. Phase1 consists of
indices0..3, all four G/P/S/B AWGN10 primary endpoints. Phase2 is indices4..83,
the remaining80 identity/AWGNinteger6..18/Rayleigheven6..18 endpoints. Within
each phase an array concurrency1 limits GPU use. Phase2 may be submitted with
afterok of the entire phase1 array, with no pruning or AP-based selection.
Indices route directly to the fixed plan, never choose weights or data subsets.
Every endpoint processes all3769 validation pairs with fixedepoch3 seed17.

Do not submit/run the GPU arrays until all44544 records have passed native AND
local007 with identical complete ledgers,11136 paired steps/33408 comparisons,
normal original rawjob/.0 and normal native-audit/export jobs/.0, plus actual
local stream pipefail exit0. Native audit queued11425240 is not a pass. Use a
separately retained complete-training closure under the runtime gate path
data/provenance/cost-field-full-native-local-closure-012.json. Its construction
must follow actual executions; no placeholder may admit inference. Runtime
checks require normal audit/export accounting again and exact proof hashes.

Prepared runner validates all frozen wrapper inputs, full84 plan membership,
runtime closure, both canonical full proofs, unchanged complete training report,
and actual accounting before importing CUDA-dependent inference. Reserve
RTX PRO6000 in enginf,4CPU/64GiB/8h per endpoint; existing pinned environment,
framework overlay005 and CUDA12.8. Assert the actual device and free memory
exceeds complete training's peak_reserved_bytes by2GiB before model loading.
No concurrent-device injection or interruption of either existing GPU job.

Run unchanged validation_code_008/infer_GPU.py with the routed arm/channel/SNR
and canonical native/local proof paths. Each output directory remains exclusive
and actual completed rawjob/.0 is required before native/local full3769 writer/
wire replay and sheng official AP. Existing separate evaluator008 and auditor008
remain unchanged. Do not label successful inference as full AP closure. Complete
original baselines/physical radio/actual resource comparisons and manuscript
remain necessary. Single seed gives descriptive results only.

Before deployment, a local CPU-only admission test of index0 must reject the
currently missing complete-training closure, exit nonzero without importing
CUDA inference or creating any endpoint output. Retain the failure log and
record this as premature-admission rejection only, not an inference test.
