#!/usr/bin/env bash
#SBATCH --job-name=paper6-KITTI-data
#SBATCH --account=engdes
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/nfs2/engdes/wc296/paper6/logs/artemis-data-deploy-001-%j.out
#SBATCH --error=/mnt/nfs2/engdes/wc296/paper6/logs/artemis-data-deploy-001-%j.err
set -euo pipefail
cd /mnt/nfs2/engdes/wc296/paper6
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
srun --ntasks=1 python3 experiments/artemis-bootstrap/deploy_data.py
