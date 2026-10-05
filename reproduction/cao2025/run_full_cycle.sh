#!/usr/bin/env bash
set -euo pipefail
cd /home/sheng/paper6
source scripts/sheng_env.sh
task_run_id="${1:?provide unique run ID}"
task_mode="${2:-formal}"
[[ "$task_run_id" =~ ^[a-z0-9-]+$ ]] || exit 2
[[ "$task_mode" == formal || "$task_mode" == engineering ]] || exit 2
task_common=(--spynet /mnt/d/paper6/checkpoints/spynet_sintel_final-3d2a1287.pth
    --train-records data/engineering/cao2025-roi-train-001.jsonl
    --train-audit data/engineering/cao2025-roi-train-audit-001.json
    --holdout-records data/engineering/cao2025-roi-holdout-001.jsonl
    --holdout-audit data/engineering/cao2025-roi-holdout-audit-001.json
    --fold data/internal-tuning-fold-001.json --protocol reproduction/cao2025/formal-protocol.md)
task_stage_end=5
if [[ "$task_mode" == engineering ]]; then task_stage_end=1; fi
for ((task_stage=1;task_stage<=task_stage_end;task_stage++)); do
    task_stage_id="${task_run_id}-stage${task_stage}"
    task_manifest="data/runs/${task_stage_id}.json"
    task_audit="data/runs/${task_stage_id}-audit.json"
    task_parent=();task_engineering=();task_engineering_audit=()
    if [[ "$task_mode" == engineering ]]; then
        task_engineering=(--engineering-steps 3);task_engineering_audit=(--engineering)
    fi
    if (( task_stage > 1 )); then
        task_previous=$((task_stage-1))
        task_parent=(--predecessor-manifest "data/runs/${task_run_id}-stage${task_previous}.json"
            --predecessor-audit "data/runs/${task_run_id}-stage${task_previous}-audit.json")
    fi
    python reproduction/cao2025/train_stages.py --stage "$task_stage" "${task_common[@]}" "${task_parent[@]}" "${task_engineering[@]}" \
        --data-root /mnt/d/paper6/data/kitti --output-dir "/mnt/d/paper6/runs/${task_stage_id}" --manifest "$task_manifest" \
        > "logs/${task_stage_id}-train.log" 2>&1
    python reproduction/cao2025/audit_training.py "${task_common[@]}" "${task_parent[@]}" "${task_engineering_audit[@]}" \
        --manifest "$task_manifest" --output "$task_audit" > "logs/${task_stage_id}-audit.log" 2>&1
    echo "Finished independently audited original variant stage${task_stage}: ${task_stage_id}"
done
