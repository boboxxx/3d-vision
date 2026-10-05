#!/usr/bin/env bash
#SBATCH --job-name=paper6-ecsic-final-audit
#SBATCH --account=engdes
#SBATCH --partition=short
#SBATCH --array=4,5%1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/nfs2/engdes/wc296/paper6/logs/ecsic-final-audit013-%j.out
#SBATCH --error=/mnt/nfs2/engdes/wc296/paper6/logs/ecsic-final-audit013-%j.err
set -euo pipefail
cd /mnt/nfs2/engdes/wc296/paper6
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
srun --ntasks=1 envs/torch-cu128-001/bin/python experiments/original-paper-scenarios/ecsic-final-candidates-code-013/native.py
