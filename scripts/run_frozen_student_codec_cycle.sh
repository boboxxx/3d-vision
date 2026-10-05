#!/usr/bin/env bash
set -euo pipefail
cd /home/sheng/paper6
source scripts/sheng_env.sh
task_run_id="${1:?provide unique run ID}"
task_config="configs/tuning/frozen_student_uniform_seed17_epoch1.yaml"
task_protocol="experiments/geometry-link/frozen-student-codec-warmup.md"
task_initialization="/mnt/d/paper6/runs/liga/home_sheng_paper6_configs_tuning/student_task_seed17_epoch1.student-task-seed17-001/ckpt/checkpoint_epoch_1.pth"
task_source_manifest="data/provenance/server-source-manifest-${task_run_id}.json"
task_output="/mnt/d/paper6/runs/liga/home_sheng_paper6_configs_tuning/frozen_student_uniform_seed17_epoch1.${task_run_id}"
task_checkpoint="${task_output}/ckpt/checkpoint_epoch_1.pth"

check_frozen_sources() {
python - "$task_source_manifest" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,'src')
from geocomm.evidence import source_identity
saved=json.loads(Path(sys.argv[1]).read_text())
trees={'project':(Path('.'),['src','scripts','configs','pyproject.toml']),
       'liga':(Path('third_party/LIGA-Stereo'),['liga','configs','tools','setup.py']),
       'mmdet':(Path('third_party/mmdetection_kitti'),['mmdet'])}
for name,(root,paths) in trees.items():
    if source_identity(root,paths)!=saved[name]['actual_sources']:
        raise RuntimeError('sources changed during F3: '+name)
PY
}
check_frozen_sources
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 scripts/run_liga.py train \
    --seed 17 --init-ckpt "$task_initialization" --codec-only --freeze-student --protocol "$task_protocol" \
    --manifest "data/runs/${task_run_id}.json" --cfg_file "$task_config" \
    --batch_size 1 --workers 4 --epochs 1 --launcher pytorch --exp_name "$task_run_id" \
    --save_to_file > "logs/${task_run_id}-train.log" 2>&1
check_frozen_sources
python scripts/audit_codec_checkpoint.py --manifest "data/runs/${task_run_id}.json" \
    --initialization "$task_initialization" --checkpoint "$task_checkpoint" --expected-steps 3340 \
    --output "data/runs/${task_run_id}-checkpoint-audit.json" > "logs/${task_run_id}-checkpoint-audit.log" 2>&1
python scripts/summarize_training.py --manifest "data/runs/${task_run_id}.json" \
    --events "$task_output/tensorboard" --expected-steps 3340 --output "data/runs/${task_run_id}-scalar-audit.json" \
    > "logs/${task_run_id}-scalar-audit.log" 2>&1
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 scripts/run_liga.py test \
    --seed 17 --manifest "data/runs/${task_run_id}-AP.json" --protocol "$task_protocol" \
    --cfg_file "$task_config" --ckpt "$task_checkpoint" --batch_size 1 --workers 4 \
    --launcher pytorch --save_to_file --eval_tag "$task_run_id" > "logs/${task_run_id}-AP.log" 2>&1
check_frozen_sources
python scripts/audit_liga_run.py --manifest "data/runs/${task_run_id}-AP.json" \
    --source-manifest "$task_source_manifest" --preparation data/kitti-preparation-001.json \
    --tuning-fold data/internal-tuning-fold-001.json \
    --output "data/runs/${task_run_id}-AP-audit.json" > "logs/${task_run_id}-AP-audit.log" 2>&1
echo "F3 cycle finished: ${task_run_id}"
