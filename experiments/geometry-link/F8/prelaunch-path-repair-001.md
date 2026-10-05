# Prelaunch eligibility path repair

The six CPU checks passed on sheng, but the following eligibility check failed
before launching any native GPU process: the file-relative root used parents[3]
and resolved to `paper6/experiments`, not `paper6`. Full error and source hash
are retained in `logs/F8-sheng-CPU-002.log` and the prelaunch failure record.
There were zero F8 native updates and no measured AP or native loss.

Correct eligibility.py to parents[4]; the runners correctly use HERE.parents[3]
and stay unchanged in that respect. Add a CPU regression that checks this root
against the already correct runner root. Rerun seven CPU checks locally and on
sheng as gate003, then run the real eligibility check. Keep prior passed gates
and the failed prelaunch check. No data, weights, native objective, seeds,
training budget, parameter scope or statistical criterion changes.
