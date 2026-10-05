#!/usr/bin/env bash
set -euo pipefail
cd /home/sheng/paper6
source scripts/sheng_env.sh
task_run_id="${1:?provide unique run ID}"
task_source="data/provenance/server-source-manifest-${task_run_id}.json"
task_checkpoint="/mnt/d/paper6/runs/liga/home_sheng_paper6_configs_tuning/frozen_student_uniform_seed17_epoch1.frozen-student-uniform-seed17-001/ckpt/checkpoint_epoch_1.pth"
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 scripts/run_liga.py test \
    --seed 17 --manifest "data/runs/${task_run_id}.json" \
    --protocol experiments/geometry-link/frozen-codec-identity-diagnostic.md \
    --cfg_file configs/diagnostic/frozen_codec_identity_holdout.yaml --ckpt "$task_checkpoint" \
    --batch_size 1 --workers 4 --launcher pytorch --save_to_file --eval_tag "$task_run_id" \
    > "logs/${task_run_id}-AP.log" 2>&1
python scripts/audit_liga_run.py --manifest "data/runs/${task_run_id}.json" \
    --source-manifest "$task_source" --preparation data/kitti-preparation-001.json \
    --tuning-fold data/internal-tuning-fold-001.json --identity-codec-diagnostic \
    --output "data/runs/${task_run_id}-audit.json" > "logs/${task_run_id}-audit.log" 2>&1
echo "Identity codec diagnostic finished: ${task_run_id}"
