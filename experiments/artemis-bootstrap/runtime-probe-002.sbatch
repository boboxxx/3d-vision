#!/usr/bin/env bash
#SBATCH --job-name=paper6-6000pro-runtime
#SBATCH --account=engdes
#SBATCH --partition=enginf
#SBATCH --gres=gpu:RTX:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/nfs2/engdes/wc296/paper6/logs/artemis-bootstrap-002-%j.out
#SBATCH --error=/mnt/nfs2/engdes/wc296/paper6/logs/artemis-bootstrap-002-%j.err
set -euo pipefail
cd /mnt/nfs2/engdes/wc296/paper6
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
srun --ntasks=1 envs/torch-cu128-001/bin/python experiments/artemis-bootstrap/probe.py \
  --output data/engineering/artemis-RTX-PRO6000-bootstrap-002.json
