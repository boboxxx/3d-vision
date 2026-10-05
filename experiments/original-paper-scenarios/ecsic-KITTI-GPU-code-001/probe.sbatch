#!/usr/bin/env bash
#SBATCH --job-name=paper6-ecsic-GPU-001
#SBATCH --account=engdes
#SBATCH --partition=enginf
#SBATCH --gres=gpu:RTX:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/nfs2/engdes/wc296/paper6/logs/ecsic-KITTI-GPU-001-%j.out
#SBATCH --error=/mnt/nfs2/engdes/wc296/paper6/logs/ecsic-KITTI-GPU-001-%j.err
set -euo pipefail
source /etc/profile.d/lmod.sh
module load gnu12/12.2.0
cd /mnt/nfs2/engdes/wc296/paper6
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 WANDB_MODE=disabled
export PYTHONPATH=/mnt/nfs2/engdes/wc296/paper6/dependencies/artemis-framework-overlay-005/site-packages
srun --ntasks=1 envs/torch-cu128-001/bin/python experiments/original-paper-scenarios/ecsic-KITTI-GPU-code-001/probe.py
