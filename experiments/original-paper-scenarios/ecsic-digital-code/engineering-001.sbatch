#!/bin/bash
#SBATCH --job-name=p6-ecsic-digital-001
#SBATCH --account=engdes
#SBATCH --qos=normal
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/nfs2/engdes/wc296/paper6/logs/artemis-ecsic-digital-CPU-001-%j.out
#SBATCH --error=/mnt/nfs2/engdes/wc296/paper6/logs/artemis-ecsic-digital-CPU-001-%j.err
set -euo pipefail
cd /mnt/nfs2/engdes/wc296/paper6
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
PYTHON=/mnt/nfs2/engdes/wc296/paper6/envs/digital-cpu-001/bin/python
"$PYTHON" experiments/original-paper-scenarios/ecsic-digital-code/run_artemis.py \
  --output data/engineering/artemis-ecsic-digital-CPU-001.json
