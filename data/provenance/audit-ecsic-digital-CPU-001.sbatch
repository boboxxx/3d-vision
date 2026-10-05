#!/bin/bash
#SBATCH --job-name=p6-ecsic-digital-audit-001
#SBATCH --account=engdes
#SBATCH --qos=normal
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/nfs2/engdes/wc296/paper6/logs/artemis-ecsic-digital-audit-001-%j.out
#SBATCH --error=/mnt/nfs2/engdes/wc296/paper6/logs/artemis-ecsic-digital-audit-001-%j.err
set -euo pipefail
cd /mnt/nfs2/engdes/wc296/paper6
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
/mnt/nfs2/engdes/wc296/paper6/envs/digital-cpu-001/bin/python \
  data/provenance/verify-ecsic-digital-CPU-001.py \
  --terminal-json data/provenance/artemis-ecsic-digital-CPU-001-terminal.json \
  --output data/provenance/ecsic-digital-CPU-001-audit.json
