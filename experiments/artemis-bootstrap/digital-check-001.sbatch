#!/usr/bin/env bash
#SBATCH --job-name=paper6-LDPC-check
#SBATCH --account=engdes
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/nfs2/engdes/wc296/paper6/logs/artemis-digital-check-001-%j.out
#SBATCH --error=/mnt/nfs2/engdes/wc296/paper6/logs/artemis-digital-check-001-%j.err
set -euo pipefail
cd /mnt/nfs2/engdes/wc296/paper6
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
srun --ntasks=1 envs/digital-cpu-001/bin/python experiments/original-paper-scenarios/digital-code/check_cpu.py
