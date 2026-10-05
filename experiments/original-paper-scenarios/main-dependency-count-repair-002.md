# Resume the completed main proof transfer, without repeating detector work

Driver001 stops after both native closure and its transfer succeed. Its main
artifact count45279 omits the24 metric/evaluator files that its own mapping
and local verifier already require. Native closure contains51 root records,
45228 predictions and24 metric/evaluator files:45303, not45279. Engineering
still has51+24=75 because it does not produce full-split AP metric files.

Preserve original driver, failed state and logs. Isolated002 fixes this counting
assertion in transfer planning and local verification, with full exact expected
artifact set unchanged. Verify old driver terminal/failed state and actual
sealed native closure identity/PID absent, then resume transfers and full
45228-prediction local verification. Do not repeat any24 detector/audit command,
rerun native closure or change any metrics/tolerance/source. After complete
proof and free-GPU checks, use unchanged SRCNNformal once-only launcher.
