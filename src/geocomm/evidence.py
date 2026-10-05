import hashlib
import json
from pathlib import Path
import subprocess
import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def serializable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError("Not serializable: " + type(value).__name__)


def save_link_accounting(path, batch_dict):
    if "communication_accounting" not in batch_dict:
        return
    row = dict(batch_dict["communication_accounting"])
    row["frame_id"] = batch_dict.get("frame_id")
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, default=serializable, allow_nan=False) + "\n")


def prediction_link_accounting(path, predictions, batch):
    """Returned sensor metadata survives DDP rebuilding the input dict."""
    recorded = predictions[0].get('batch_dict', batch) if predictions else batch
    save_link_accounting(path, recorded)


def revision(root):
    p = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                       capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def source_identity(root, folders):
    """Hash deployed sources even when rsync intentionally omits Git metadata."""
    root = Path(root)
    files = {}
    for folder in folders:
        entry = root / folder
        paths = [entry] if entry.is_file() else entry.rglob("*")
        for path in paths:
            if (path.is_file() and path.suffix in {".py", ".cpp", ".cu", ".h", ".yaml", ".toml"}
                    and "__pycache__" not in path.parts and "build" not in path.parts
                    and path.name != "version.py"):
                files[str(path.relative_to(root))] = sha256(path)
    manifest = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return dict(sha256=hashlib.sha256(manifest).hexdigest(), file_hashes=files,
                generated_version_py_excluded=True)
