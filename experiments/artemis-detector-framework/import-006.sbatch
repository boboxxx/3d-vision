#!/usr/bin/env bash
#SBATCH --job-name=paper6-framework-import
#SBATCH --account=engdes
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/nfs2/engdes/wc296/paper6/logs/artemis-framework-import-006-%j.out
#SBATCH --error=/mnt/nfs2/engdes/wc296/paper6/logs/artemis-framework-import-006-%j.err
set -euo pipefail
source /etc/profile.d/lmod.sh
module load gnu12/12.2.0
cd /mnt/nfs2/engdes/wc296/paper6
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
srun --ntasks=1 envs/torch-cu128-001/bin/python experiments/artemis-detector-framework/import-006.py
