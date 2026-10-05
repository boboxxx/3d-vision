#!/bin/bash
#SBATCH --job-name=p6-framing-pixels
#SBATCH --account=engdes
#SBATCH --qos=normal
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/nfs2/engdes/wc296/paper6/logs/artemis-digital-framing-pixels-001-%j.out
#SBATCH --error=/mnt/nfs2/engdes/wc296/paper6/logs/artemis-digital-framing-pixels-001-%j.err
set -euo pipefail
cd /mnt/nfs2/engdes/wc296/paper6
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
srun --ntasks=1 envs/digital-cpu-001/bin/python data/provenance/capture-framing-codec-pixels-001.py
