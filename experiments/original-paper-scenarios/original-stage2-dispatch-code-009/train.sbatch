#!/usr/bin/env bash
#SBATCH --job-name=paper6-original-public30-stage2
#SBATCH --account=engdes
#SBATCH --partition=enginf
#SBATCH --gres=gpu:RTX:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/nfs2/engdes/wc296/paper6/logs/original-public30-stage2-009-%j.out
#SBATCH --error=/mnt/nfs2/engdes/wc296/paper6/logs/original-public30-stage2-009-%j.err
set -euo pipefail
source /etc/profile.d/lmod.sh
module load gnu12/12.2.0
cd /mnt/nfs2/engdes/wc296/paper6
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
srun --ntasks=1 envs/torch-cu128-001/bin/python experiments/original-paper-scenarios/original-stage2-dispatch-code-009/run.py
