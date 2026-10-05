# F9 terminal evidence storage path clarification

Locked before running terminal closure or interpreting complete F9 results.
The running experiment and its eight frozen source trees remain unchanged.
Read-only inspection of the completed U-awgn10 manifest shows metrics and
predictions under /mnt/d/paper6/runs, rather than /home/sheng/paper6. Prepared
closure001 assumes every transferred file is below the checkout; prepared
local verifier001 has the same assumption. Neither has executed. Preserve both.

Use closure002/verifier002 with only evidence storage mapping changed. Native
checkout files retain their existing relative local paths. Native D: metrics,
result pickle, communication records and prediction texts map to local
data/engineering/F9-formal-native-transfer-001/<absolute-native-path-without-/>
and retain native-path-to-local-path mappings and freshly verified SHA256.
Transfer those D: files from the remote filesystem root with a separate exact
files-from list. Keep all complete-row, source, checkpoint, actual terminal,
paired noise/input, official AP, feature and prelocked-difference checks.
No model, data, condition, seed, optimization, RNG, AP computation or result
selection changes. Larger weights remain native and fully audited there.
