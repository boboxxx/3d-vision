# Existing CUDA math-header search paths, build005

Locked before compilation. Actual CPU Slurm11424741 ended FAILED1:0 after
unchanged MMCV C++ compilation reached CUDA: ATen/CUDAContextLight.h requires
cusparse.h, cublas_v2.h, cublasLt.h and cusolverDn.h. The minimal toolkit lacks
these math headers. Existing frozen torch-runtime NVIDIA wheels already contain
them; no new toolkit/library installation is required. Preserve all logs,
overlay004 and the failed report, which confirms unchanged runtime/sources.

Add only the four existing NVIDIA include directories (cublas/cusparse/cusolver/
curand) to CPATH and CPLUS_INCLUDE_PATH. Lock every actual header SHA in input005
before execution and recheck before/after. Leave MMCV sources and CUDA libraries
unchanged. Build a tiny compile-only probe that includes ATen/cuda/CUDAContext.h
with explicit sm120/C++17 and Torch includes; if it fails, stop before repeating
the full build. The probe performs no GPU execution or mathematics.

Then compile unchanged MMCV from the same locked archive into independent
overlay005, real-import probe005, CPU Slurm with the same resource limits and
offline cache004. All prior runtime/source/terminal and later sensor-only
GPU/task/teacher/backward boundaries remain. A header-path fix is not evidence
that framework compatibility, detector forward or AP will pass.
