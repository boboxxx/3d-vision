"""Install into one unique project venv on allocated CPU compute only."""
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    target = root / "envs/torch-cu128-001"
    output = root / "data/engineering/artemis-runtime-setup-001.json"
    if not os.environ.get("SLURM_JOB_ID") or target.exists() or output.exists():
        raise RuntimeError("allocated CPU job and unique environment/evidence required")
    target.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {"state": "starting", "job_id": os.environ["SLURM_JOB_ID"],
              "started_at_unix": time.time(), "environment": str(target),
              "protocol_sha256": hashlib.sha256((root / "experiments/artemis-bootstrap/runtime-protocol-001.md").read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with output.open("x") as handle:
        json.dump(record, handle, indent=2)
    try:
        subprocess.run([sys.executable, "-m", "venv", str(target)], check=True)
        python = str(target / "bin/python")
        report1 = root / "data/engineering/artemis-runtime-torch-wheels-001.json"
        report2 = root / "data/engineering/artemis-runtime-base-wheels-001.json"
        subprocess.run([python, "-m", "pip", "install", "--no-cache-dir", "--report", str(report1),
                        "torch==2.7.1", "--index-url", "https://download.pytorch.org/whl/cu128"], check=True)
        subprocess.run([python, "-m", "pip", "install", "--no-cache-dir", "--report", str(report2),
                        "numpy==1.26.4", "Pillow==10.2.0", "PyYAML==6.0.2"], check=True)
        subprocess.run([python, "-m", "pip", "check"], check=True)
        freeze = subprocess.check_output([python, "-m", "pip", "freeze"], text=True)
        freeze_path = root / "data/engineering/artemis-runtime-freeze-001.txt"
        with freeze_path.open("x") as handle:
            handle.write(freeze)
        record["freeze_sha256"] = hashlib.sha256(freeze.encode()).hexdigest()
        record["wheel_hashes"] = [
            {"name": item["metadata"]["name"], "version": item["metadata"]["version"],
             "hashes": item["download_info"]["archive_info"].get("hashes", {})}
            for report in (report1, report2) for item in json.loads(report.read_text())["install"]]
        cpu = subprocess.check_output([python, "-c", "import json,torch,numpy,PIL,yaml; print(json.dumps(dict(torch=torch.__version__,CUDA=torch.version.cuda,numpy=numpy.__version__,Pillow=PIL.__version__,PyYAML=yaml.__version__)))"], text=True)
        record["CPU_imports"] = json.loads(cpu)
        if record["CPU_imports"]["torch"] != "2.7.1+cu128" or record["CPU_imports"]["CUDA"] != "12.8":
            raise RuntimeError("runtime wheel identity mismatch")
        record["state"] = "installed_CPU_imports_and_pip_check_passed_GPU_not_tested"
    except Exception as error:
        record["state"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["finished_at_unix"] = time.time()
        output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
