#!/usr/bin/env bash
# Fixed F1 cycle: five full feature epochs, saved-state audit, native 3D AP,
# independent prediction/GT/AP audit. Stage failures preserve all evidence.
set -euo pipefail
cd /home/sheng/paper6
source scripts/sheng_env.sh
task_run_id="${1:?provide a unique run ID}"
task_source_manifest="data/provenance/server-source-manifest-${task_run_id}.json"
task_run_output="/mnt/d/paper6/runs/${task_run_id}"
task_config="configs/tuning/student_feature_seed17_epoch5.yaml"
task_protocol="experiments/geometry-link/student-warmup.md"
task_initialization="/mnt/d/paper6/checkpoints/liga-author-download"

check_frozen_sources() {
    python - "$task_source_manifest" <<'PY'
import json
from pathlib import Path
import sys
sys.path.insert(0, 'src')
from geocomm.evidence import source_identity
saved = json.loads(Path(sys.argv[1]).read_text())
trees = {'project': (Path('.'), ['src', 'scripts', 'configs', 'pyproject.toml']),
         'liga': (Path('third_party/LIGA-Stereo'), ['liga', 'configs', 'tools', 'setup.py']),
         'mmdet': (Path('third_party/mmdetection_kitti'), ['mmdet'])}
for name, (root, paths) in trees.items():
    if source_identity(root, paths) != saved[name]['actual_sources']:
        raise RuntimeError('sources changed during fixed F1 cycle: '+name)
PY
}

check_frozen_sources
python scripts/pretrain_student.py --config "$task_config" --checkpoint "$task_initialization" \
    --protocol "$task_protocol" --output-dir "$task_run_output" \
    --manifest "data/runs/${task_run_id}.json" > "logs/${task_run_id}-train.log" 2>&1
check_frozen_sources
python scripts/audit_student_pretraining.py --manifest "data/runs/${task_run_id}.json" \
    --initialization "$task_initialization" --source-manifest "$task_source_manifest" \
    --config "$task_config" --protocol "$task_protocol" --fold data/internal-tuning-fold-001.json \
    --output "data/runs/${task_run_id}-training-audit.json" > "logs/${task_run_id}-training-audit.log" 2>&1
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 scripts/run_liga.py test \
    --seed 17 --manifest "data/runs/${task_run_id}-AP.json" --protocol "$task_protocol" \
    --cfg_file "$task_config" --ckpt "$task_run_output/checkpoint_epoch_5.pth" \
    --batch_size 1 --workers 4 --launcher pytorch --save_to_file --eval_tag "$task_run_id" \
    > "logs/${task_run_id}-AP.log" 2>&1
check_frozen_sources
python scripts/audit_liga_run.py --manifest "data/runs/${task_run_id}-AP.json" \
    --source-manifest "$task_source_manifest" --preparation data/kitti-preparation-001.json \
    --tuning-fold data/internal-tuning-fold-001.json --diagnostic-holdout \
    --output "data/runs/${task_run_id}-AP-audit.json" > "logs/${task_run_id}-AP-audit.log" 2>&1
echo "F1 cycle finished: ${task_run_id}"
