#!/usr/bin/env bash
#SBATCH --job-name=paper6-LIGA-cost-field-opt
#SBATCH --account=engdes
#SBATCH --partition=enginf
#SBATCH --gres=gpu:RTX:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/nfs2/engdes/wc296/paper6/logs/cost-field-GPU-optimizer-003-%j.out
#SBATCH --error=/mnt/nfs2/engdes/wc296/paper6/logs/cost-field-GPU-optimizer-003-%j.err
set -euo pipefail
source /etc/profile.d/lmod.sh
module load gnu12/12.2.0
cd /mnt/nfs2/engdes/wc296/paper6
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=/mnt/nfs2/engdes/wc296/paper6/dependencies/artemis-framework-overlay-005/site-packages
export CUDA_HOME=/mnt/nfs2/engdes/wc296/paper6/toolchains/cuda-12.8-001
srun --ntasks=1 envs/torch-cu128-001/bin/python experiments/cost-field/code_003/probe_GPU.py
