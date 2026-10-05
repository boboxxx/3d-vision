#!/usr/bin/env bash
# Source inside the dedicated sheng checkout. Shared conda packages stay intact.
task_project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_HOME="${CUDA_HOME:-/home/sheng/anaconda3/envs/epiu-dsgn}"
export PATH="$task_project_dir/.venv/bin:$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=2
export OMP_NUM_THREADS=4
export NUMBA_NUM_THREADS=4
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
if [[ -f /usr/lib/wsl/lib/libcuda.so.1 ]]; then
    export NUMBA_CUDA_DRIVER="$(python "$task_project_dir/scripts/cuda_driver_path.py")"
fi
unset task_project_dir
