#!/usr/bin/env bash
set -euo pipefail
cd /home/sheng/paper6
source scripts/sheng_env.sh
task_run_id="${1:?provide unique run ID}"
task_checkpoint="/mnt/d/paper6/runs/liga/home_sheng_paper6_configs_tuning/frozen_student_uniform_seed17_epoch1.frozen-student-uniform-seed17-001/ckpt/checkpoint_epoch_1.pth"
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 scripts/diagnose_codec_features.py \
    --feature-output "data/runs/${task_run_id}-features.jsonl" test --seed 17 \
    --manifest "data/runs/${task_run_id}.json" --protocol experiments/geometry-link/frozen-codec-feature-diagnostic.md \
    --cfg_file configs/diagnostic/frozen_codec_identity_holdout.yaml --ckpt "$task_checkpoint" \
    --batch_size 1 --workers 4 --launcher pytorch --save_to_file --eval_tag "$task_run_id" \
    > "logs/${task_run_id}-AP.log" 2>&1
python scripts/audit_liga_run.py --manifest "data/runs/${task_run_id}.json" \
    --source-manifest "data/provenance/server-source-manifest-${task_run_id}.json" \
    --preparation data/kitti-preparation-001.json --tuning-fold data/internal-tuning-fold-001.json \
    --identity-codec-diagnostic --output "data/runs/${task_run_id}-AP-audit.json" \
    > "logs/${task_run_id}-AP-audit.log" 2>&1
python scripts/audit_codec_features.py --features "data/runs/${task_run_id}-features.jsonl" \
    --manifest "data/runs/${task_run_id}.json" --AP-audit "data/runs/${task_run_id}-AP-audit.json" \
    --fold data/internal-tuning-fold-001.json --output "data/runs/${task_run_id}-feature-audit.json" \
    > "logs/${task_run_id}-feature-audit.log" 2>&1
echo "Codec feature diagnostic finished: ${task_run_id}"
