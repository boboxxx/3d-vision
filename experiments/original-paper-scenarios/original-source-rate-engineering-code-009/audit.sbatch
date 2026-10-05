#!/usr/bin/env bash
#SBATCH --job-name=paper6-source-audit009
#SBATCH --account=engdes
#SBATCH --partition=short
#SBATCH --array=10,50%1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/nfs2/engdes/wc296/paper6/logs/original-source-audit009-%j.out
#SBATCH --error=/mnt/nfs2/engdes/wc296/paper6/logs/original-source-audit009-%j.err
set -euo pipefail
source /etc/profile.d/lmod.sh
module load gnu12/12.2.0
cd /mnt/nfs2/engdes/wc296/paper6
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
srun --ntasks=1 envs/torch-cu128-001/bin/python experiments/original-paper-scenarios/original-source-rate-engineering-code-009/audit.py "$@"
