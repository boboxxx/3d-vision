#!/usr/bin/env bash
#SBATCH --job-name=paper6-6000pro-probe
#SBATCH --account=engdes
#SBATCH --partition=enginf
#SBATCH --gres=gpu:RTX:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/nfs2/engdes/wc296/paper6/logs/artemis-bootstrap-001-%j.out
#SBATCH --error=/mnt/nfs2/engdes/wc296/paper6/logs/artemis-bootstrap-001-%j.err

set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd /mnt/nfs2/engdes/wc296/paper6
srun --ntasks=1 python3 experiments/artemis-bootstrap/probe.py \
  --output data/engineering/artemis-RTX-PRO6000-bootstrap-001.json
