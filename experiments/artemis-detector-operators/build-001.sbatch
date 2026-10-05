#!/usr/bin/env bash
#SBATCH --job-name=paper6-native-sm120-build
#SBATCH --account=engdes
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/nfs2/engdes/wc296/paper6/logs/artemis-detector-operators-build-001-%j.out
#SBATCH --error=/mnt/nfs2/engdes/wc296/paper6/logs/artemis-detector-operators-build-001-%j.err
set -euo pipefail
source /etc/profile.d/lmod.sh
module load gnu12/12.2.0
cd /mnt/nfs2/engdes/wc296/paper6
export OMP_NUM_THREADS=4
srun --ntasks=1 envs/torch-cu128-001/bin/python experiments/artemis-detector-operators/code/build.py
