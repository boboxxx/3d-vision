#!/usr/bin/env bash
#SBATCH --job-name=paper6-native-sm120-probe
#SBATCH --account=engdes
#SBATCH --partition=enginf
#SBATCH --gres=gpu:RTX:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/nfs2/engdes/wc296/paper6/logs/artemis-detector-operators-GPU-001-%j.out
#SBATCH --error=/mnt/nfs2/engdes/wc296/paper6/logs/artemis-detector-operators-GPU-001-%j.err
set -euo pipefail
cd /mnt/nfs2/engdes/wc296/paper6
export OMP_NUM_THREADS=4
srun --ntasks=1 envs/torch-cu128-001/bin/python experiments/artemis-detector-operators/code/probe.py
