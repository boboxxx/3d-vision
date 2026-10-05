#!/usr/bin/env bash
set -euo pipefail
cd /home/sheng/paper6
source scripts/sheng_env.sh
task_run_id="${1:?provide unique run ID}"
task_initialization="/mnt/d/paper6/runs/stereo-feature-native-seed17-001/checkpoint_epoch_1.pth"
task_config="configs/tuning/stereo_task_seed17_epoch1.yaml"
task_protocol="${2:-experiments/geometry-link/stereo-native-task-adaptation.md}"
task_fold="data/internal-tuning-fold-001.json"
task_source="data/provenance/server-source-manifest-${task_run_id}.json"
task_output="/mnt/d/paper6/runs/${task_run_id}"
task_checkpoint="${task_output}/checkpoint_epoch_1.pth"

check_sources() {
python - "$task_source" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,'src')
from geocomm.evidence import source_identity
saved=json.loads(Path(sys.argv[1]).read_text())
for name,(root,paths) in {'project':(Path('.'),['src','scripts','configs','pyproject.toml']),
        'liga':(Path('third_party/LIGA-Stereo'),['liga','configs','tools','setup.py']),
        'mmdet':(Path('third_party/mmdetection_kitti'),['mmdet']),
        'stereo_rcnn':(Path('third_party/Stereo-RCNN'),['lib','demo.py','test_net.py'])}.items():
    if source_identity(root,paths)!=saved[name]['actual_sources']:
        raise RuntimeError('F6 source changed: '+name)
PY
}

check_sources
python scripts/adapt_stereo_native_task.py --config "$task_config" --checkpoint "$task_initialization" \
    --protocol "$task_protocol" --fold "$task_fold" --source-manifest "$task_source" \
    --output-dir "$task_output" --manifest "data/runs/${task_run_id}.json" \
    > "logs/${task_run_id}-train.log" 2>&1
check_sources
python scripts/audit_stereo_native_task.py --manifest "data/runs/${task_run_id}.json" \
    --initialization "$task_initialization" --config "$task_config" --protocol "$task_protocol" \
    --fold "$task_fold" --source-manifest "$task_source" --output "data/runs/${task_run_id}-training-audit.json" \
    > "logs/${task_run_id}-training-audit.log" 2>&1
task_checkpoint_hash="$(sha256sum "$task_checkpoint" | cut -d ' ' -f1)"
for task_channel in identity awgn; do
    task_eval_config="configs/diagnostic/stereo_feature_${task_channel}_holdout.yaml"
    task_eval_id="${task_run_id}-${task_channel}"
    python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 scripts/diagnose_stereo_features.py \
        --records "data/runs/${task_eval_id}-features.jsonl" --report "data/runs/${task_eval_id}-boundary-report.json" \
        --checkpoint-sha256 "$task_checkpoint_hash" --expected-channel "$task_channel" \
        test --seed 17 --manifest "data/runs/${task_eval_id}.json" --protocol "$task_protocol" \
        --cfg_file "$task_eval_config" --ckpt "$task_checkpoint" \
        --batch_size 1 --workers 4 --launcher pytorch --save_to_file --eval_tag "$task_eval_id" \
        > "logs/${task_eval_id}-AP.log" 2>&1
    check_sources
    task_identity_flag=()
    if [[ "$task_channel" == identity ]]; then task_identity_flag=(--identity-codec-diagnostic); fi
    python scripts/audit_liga_run.py --manifest "data/runs/${task_eval_id}.json" \
        --source-manifest "$task_source" --preparation data/kitti-preparation-001.json \
        --tuning-fold "$task_fold" --stereo-feature-link "${task_identity_flag[@]}" --output "data/runs/${task_eval_id}-AP-audit.json" \
        > "logs/${task_eval_id}-AP-audit.log" 2>&1
    python scripts/audit_stereo_features.py --features "data/runs/${task_eval_id}-features.jsonl" \
        --report "data/runs/${task_eval_id}-boundary-report.json" \
        --manifest "data/runs/${task_eval_id}.json" --AP-audit "data/runs/${task_eval_id}-AP-audit.json" \
        --fold "$task_fold" --expected-channel "$task_channel" --checkpoint-sha256 "$task_checkpoint_hash" \
        --output "data/runs/${task_eval_id}-feature-audit.json" > "logs/${task_eval_id}-feature-audit.log" 2>&1
done
check_sources
echo "F6 clean native3D codec task adaptation finished: ${task_run_id}"
