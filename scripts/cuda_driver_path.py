#!/usr/bin/env python3
"""Identify WSL's concrete driver library actually loaded by working PyTorch.

The /usr/lib/wsl/lib facade can fail Numba ctypes context calls. Use the same
concrete Windows-driver ELF as the working CUDA runtime; never select a stub.
"""
from pathlib import Path
import torch

torch.cuda.init()
paths = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
         if "libcuda.so" in line and line.split()[-1].startswith("/")}
concrete = sorted(path for path in paths if path.startswith("/usr/lib/wsl/drivers/"))
if len(concrete) != 1:
    raise RuntimeError("cannot identify a unique loaded concrete WSL CUDA driver: " + repr(concrete))
print(concrete[0])
