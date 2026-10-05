"""Allocated-only GPU product/runtime inventory, no research model or dataset."""
import argparse
import csv
import hashlib
import io
import json
import os
import platform
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("GPU inventory must run in allocated Slurm step")
    if args.output.exists():
        raise RuntimeError("bootstrap evidence ID already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    record = {"state": "running", "scope": "GPU_bootstrap_only_no_training_AP",
              "started_at_unix": time.time(), "hostname": platform.node(),
              "slurm": {k: os.environ.get(k) for k in (
                  "SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_JOB_ACCOUNT", "SLURM_STEP_ID",
                  "SLURM_JOB_GPUS", "CUDA_VISIBLE_DEVICES", "SLURM_CPUS_PER_TASK")},
              "sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(here.iterdir()) if p.is_file()},
              "Python": platform.python_version()}
    with args.output.open("x") as handle:
        json.dump(record, handle, indent=2)
    try:
        cmd = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version", "--format=csv,noheader,nounits"]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
        products = [{"index": v[0].strip(), "uuid": v[1].strip(), "name": v[2].strip(),
                     "memory_MiB": int(v[3]), "driver": v[4].strip()}
                    for v in csv.reader(io.StringIO(result.stdout))]
        record["GPU_products"] = products
        logical = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if len(logical) != 1:
            raise RuntimeError("bootstrap requires exactly one allocated GPU")
        candidates = [v for v in products if v["index"] == logical[0] or v["uuid"] == logical[0]]
        if len(candidates) != 1:
            raise RuntimeError("cannot verify allocated logical GPU from nvidia-smi inventory")
        gpu = candidates[0]
        record["allocated_GPU"] = gpu
        if "RTX PRO 6000" not in gpu["name"].upper() or gpu["memory_MiB"] < 90 * 1024:
            raise RuntimeError("allocated device does not meet requested RTX PRO6000/90GiB")
        record["hardware_verified"] = True
        record["GPU_tensor_probe_passed"] = False
        try:
            import torch
            record["PyTorch"] = torch.__version__
            record["torch_CUDA"] = torch.version.cuda
            if not torch.cuda.is_available():
                raise RuntimeError("PyTorch cannot access allocated GPU")
            properties = torch.cuda.get_device_properties(0)
            record["torch_GPU"] = {"name": properties.name, "major": properties.major,
                                   "minor": properties.minor, "total_memory": properties.total_memory}
            record["torch_compiled_archs"] = torch.cuda.get_arch_list()
            arch = f"sm_{properties.major}{properties.minor}"
            if arch not in record["torch_compiled_archs"]:
                raise RuntimeError("existing PyTorch wheel lacks allocated GPU architecture; isolated new runtime required")
            torch.manual_seed(17)
            x = torch.randn(256, 256, device="cuda", requires_grad=True)
            w = torch.randn(256, 256, device="cuda", requires_grad=True)
            loss = (x @ w).square().mean()
            loss.backward()
            if not torch.isfinite(loss) or not torch.isfinite(x.grad).all() or not torch.isfinite(w.grad).all():
                raise RuntimeError("nonfinite allocated-GPU matrix gradients")
            conv = torch.nn.Conv3d(4, 8, 3, padding=1).cuda()
            volume = torch.randn(1, 4, 8, 12, 16, device="cuda", requires_grad=True)
            conv_loss = conv(volume).square().mean()
            conv_loss.backward()
            torch.cuda.synchronize()
            if not torch.isfinite(volume.grad).all() or not torch.isfinite(conv.weight.grad).all():
                raise RuntimeError("nonfinite allocated-GPU convolution gradients")
            record["GPU_tensor_probe_passed"] = True
            record["probe_losses"] = [float(loss.detach()), float(conv_loss.detach())]
            record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        except Exception as error:
            record["runtime_limitation"] = f"{type(error).__name__}: {error}"
        record["state"] = "hardware_and_tensor_probe_passed" if record["GPU_tensor_probe_passed"] else "hardware_verified_runtime_needs_setup"
    except Exception as error:
        record["state"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["finished_at_unix"] = time.time()
        args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
