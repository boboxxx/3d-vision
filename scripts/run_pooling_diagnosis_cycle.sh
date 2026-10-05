#!/usr/bin/env bash
set -euo pipefail
cd /home/sheng/paper6
source scripts/sheng_env.sh
task_run_id="${1:?provide unique run ID}"
task_source="data/provenance/server-source-manifest-${task_run_id}.json"
task_config="configs/diagnostic/student_uncompressed_holdout.yaml"
task_checkpoint="/mnt/d/paper6/runs/codec-feature-seed17-001/checkpoint_epoch_1.pth"
task_protocol="experiments/geometry-link/pooling-diagnosis.md"
check_sources() {
python - "$task_source" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,'src')
from geocomm.evidence import source_identity
saved=json.loads(Path(sys.argv[1]).read_text())
for name,(root,paths) in {'project':(Path('.'),['src','scripts','configs','pyproject.toml']),
        'liga':(Path('third_party/LIGA-Stereo'),['liga','configs','tools','setup.py']),
        'mmdet':(Path('third_party/mmdetection_kitti'),['mmdet'])}.items():
    if source_identity(root,paths)!=saved[name]['actual_sources']: raise RuntimeError('pooling suite source changed: '+name)
PY
}
for task_condition in control cost_only appearance_only both depth_preserved; do
    check_sources
    task_eval_id="${task_run_id}-${task_condition}"
    python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 scripts/diagnose_pooling.py \
        --condition "$task_condition" --records "data/runs/${task_eval_id}-pooling.jsonl" --report "data/runs/${task_eval_id}-pooling-report.json" \
        test --seed 17 --manifest "data/runs/${task_eval_id}.json" --protocol "$task_protocol" \
        --cfg_file "$task_config" --ckpt "$task_checkpoint" --batch_size 1 --workers 4 --launcher pytorch --save_to_file --eval_tag "$task_eval_id" \
        > "logs/${task_eval_id}-AP.log" 2>&1
    check_sources
    python scripts/audit_liga_run.py --manifest "data/runs/${task_eval_id}.json" --source-manifest "$task_source" \
        --preparation data/kitti-preparation-001.json --tuning-fold data/internal-tuning-fold-001.json --diagnostic-holdout \
        --output "data/runs/${task_eval_id}-AP-audit.json" > "logs/${task_eval_id}-AP-audit.log" 2>&1
    python scripts/audit_pooling.py --manifest "data/runs/${task_eval_id}.json" --report "data/runs/${task_eval_id}-pooling-report.json" \
        --records "data/runs/${task_eval_id}-pooling.jsonl" --AP-audit "data/runs/${task_eval_id}-AP-audit.json" --checkpoint "$task_checkpoint" \
        --config "$task_config" --protocol "$task_protocol" --fold data/internal-tuning-fold-001.json --source-manifest "$task_source" \
        --output "data/runs/${task_eval_id}-pooling-audit.json" > "logs/${task_eval_id}-pooling-audit.log" 2>&1
    echo "Finished full372 pooling diagnosis: ${task_eval_id}"
done
check_sources
