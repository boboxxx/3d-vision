#!/usr/bin/env bash
#SBATCH --job-name=paper6-original-PRO6000
#SBATCH --account=engdes
#SBATCH --partition=enginf
#SBATCH --gres=gpu:RTX:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/nfs2/engdes/wc296/paper6/logs/artemis-original-native-001-%j.out
#SBATCH --error=/mnt/nfs2/engdes/wc296/paper6/logs/artemis-original-native-001-%j.err
set -euo pipefail
cd /mnt/nfs2/engdes/wc296/paper6
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
srun --ntasks=1 envs/torch-cu128-001/bin/python experiments/artemis-bootstrap/verify_original_native.py
