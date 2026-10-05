"""Exclusive CPU evidence; optional single prelocked training-frame smoke."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, features

from contracts import digital_resources, normalize_symbols, require, scenarios, transmit
from image_codecs import encode_pair


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args()
    require(not args.output.exists() and not args.log.exists(), "evidence ID already exists")
    require((args.data_root is None) == (args.artifact_dir is None), "paired smoke arguments required")
    if args.artifact_dir is not None:
        require(not args.artifact_dir.exists(), "smoke artifact ID already exists")
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    protocol = here.parent / "protocol.md"
    source = {p.name: sha(p) for p in sorted(here.glob("*.py"))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    evidence = {"state": "running", "scope": "CPU_engineering_not_AP_or_LDPC_simulation",
                "started_at_unix": time.time(), "command": sys.argv,
                "protocol_sha256": sha(protocol), "source_files": source,
                "environment": {"Python": platform.python_version(), "platform": platform.platform(),
                                "NumPy": np.__version__, "Pillow": PIL.__version__,
                                "libjpeg": features.version_codec("jpg"),
                                "OpenJPEG": features.version_codec("jpg_2000")},
                "scenario_count_unique": len(scenarios()),
                "scenario_table_counts": {t: sum(t in r["tables"] for r in scenarios())
                                          for t in ("source", "channel", "joint")}}
    # Preserve failure evidence. This reservation also prevents reuse of this ID.
    with args.output.open("x") as handle:
        json.dump(evidence, handle, indent=2)
    try:
        with args.log.open("x") as handle:
            result = subprocess.run([sys.executable, "-m", "unittest", "-v", "test_contracts"],
                                    cwd=here, stdout=handle, stderr=subprocess.STDOUT)
        require(result.returncode == 0, "CPU regression failed; see preserved log")
        require("Ran 7 tests" in args.log.read_text() and "\nOK\n" in args.log.read_text(),
                "unexpected test count")
        evidence["tests_passed"] = 7
        if args.data_root is not None:
            fold_path = root / "data/internal-tuning-fold-001.json"
            fold = json.loads(fold_path.read_text())
            require("000000" in fold["folds"]["geocomm_tune_train"]["ids"], "smoke frame not training")
            paths = [args.data_root / "training" / view / "000000.png" for view in ("image_2", "image_3")]
            arrays = []
            for p in paths:
                with Image.open(p) as image:
                    require(image.mode == "RGB", "native RGB PNG required")
                    arrays.append(np.array(image, dtype=np.uint8))
            args.artifact_dir.mkdir(parents=True, exist_ok=False)
            smoke = {"frame_id": "000000", "fold_sha256": sha(fold_path),
                     "native_sources": {str(p): sha(p) for p in paths}, "codecs": []}
            for codec, parameter in (("jpeg", 50), ("jpeg2000", 10), ("jpeg2000", 30), ("jpeg2000", 50)):
                wire, _, record = encode_pair(*arrays, codec=codec, parameter=parameter)
                p = args.artifact_dir / f"{codec}-{parameter}.p6sb"
                with p.open("xb") as handle:
                    handle.write(wire)
                record["wire_path"] = str(p)
                record["resources"] = [digital_resources(
                    source_bits=record["source_bits"], n=96, k=k, qam=qam,
                    reliable_uses=0, reliable_energy=0) for k, qam in ((64, 64), (48, 256))]
                record["LDPC_identity"] = "synthetic_lengths_for_accounting_ONLY_not_reference_standard"
                smoke["codecs"].append(record)
            symbols = normalize_symbols(np.random.default_rng(17).standard_normal((4096, 2)))
            smoke["channels"] = []
            for kind in ("identity", "awgn", "rayleigh"):
                output, record = transmit(symbols, kind=kind, snr_db=10,
                                          noise_rng=np.random.default_rng(17),
                                          fading_rng=np.random.default_rng(18))
                p = args.artifact_dir / f"{kind}-10dB.npz"
                with p.open("xb") as handle:
                    np.savez(handle, transmitted=symbols, received=output)
                record["arrays_path"] = str(p)
                record["arrays_file_sha256"] = sha(p)
                smoke["channels"].append(record)
            require(all(sha(p) == smoke["native_sources"][str(p)] for p in paths), "native sources changed")
            evidence["smoke"] = smoke
        require(source == {p.name: sha(p) for p in sorted(here.glob("*.py"))}, "source changed during checks")
        require(evidence["protocol_sha256"] == sha(protocol), "protocol changed during checks")
        evidence["state"] = "passed"
    except Exception as error:
        evidence["state"] = "failed"
        evidence["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        evidence["finished_at_unix"] = time.time()
        if args.log.exists():
            evidence["log_sha256"] = sha(args.log)
        args.output.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"state": evidence["state"], "tests": evidence["tests_passed"],
                      "unique_scenarios": len(scenarios()), "output": str(args.output)}))


if __name__ == "__main__":
    main()
