"""Independently verify transferred streams and replay saved complex channels."""
import hashlib
import io
import json
import math
import struct
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/provenance/original-scenarios-CPU-verification-001.json"
assert not OUT.exists(), "do not overwrite evidence"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(array):
    descriptor = {"dtype": array.dtype.str, "shape": list(array.shape)}
    return hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode() + b"\0" +
                          np.ascontiguousarray(array).tobytes()).hexdigest()


local = json.loads((ROOT / "data/engineering/original-scenarios-local-CPU-001.json").read_text())
server_path = ROOT / "data/engineering/original-scenarios-sheng-CPU-001.json"
server = json.loads(server_path.read_text())
assert local["state"] == server["state"] == "passed"
assert local["tests_passed"] == server["tests_passed"] == 7
assert local["source_files"] == server["source_files"]
code = ROOT / "experiments/original-paper-scenarios/code"
assert server["source_files"] == {p.name: sha(p) for p in sorted(code.glob("*.py"))}
assert local["protocol_sha256"] == server["protocol_sha256"] == sha(code.parent / "protocol.md")
assert sha(ROOT / "data/engineering/original-scenarios-sheng-CPU-001.log") == server["log_sha256"]
artifacts = ROOT / "data/engineering/original-scenarios-CPU-smoke-001"
files = {}
for record in server["smoke"]["codecs"]:
    p = artifacts / Path(record["wire_path"]).name
    raw = p.read_bytes()
    assert sha(p) == record["wire_sha256"]
    magic, version, codec, reserved, nl, nr, crc = struct.unpack(">4sBBHIII", raw[:20])
    assert magic == b"P6SB" and version == 1 and reserved == 0
    assert len(raw) == 20 + nl + nr == record["source_bytes"]
    assert [nl, nr] == record["codestream_bytes"] and zlib.crc32(raw[20:]) == crc
    assert len(raw) * 8 == record["source_bits"]
    assert math.isclose(record["raw_RGB8_bits"] / record["source_bits"],
                        record["actual_raw_to_serialized_ratio"], abs_tol=1e-12)
    decode_hashes = []
    for body in (raw[20:20+nl], raw[20+nl:]):
        with Image.open(io.BytesIO(body)) as image:
            assert image.format == ("JPEG" if codec == 1 else "JPEG2000")
            array = np.array(image)
            assert array.dtype == np.uint8 and list(array.shape) == record["left_input"]["shape"]
            decode_hashes.append(array_sha(array))
    # Cross-build decoded pixels are reported; equality is not assumed by contract.
    backend_agreement = decode_hashes == [record["left_received"]["sha256"], record["right_received"]["sha256"]]
    for resources in record["resources"]:
        bits, n, k, qam = record["source_bits"], 96, resources["k"], resources["QAM"]
        nb = (bits + k - 1) // k
        m = 6 if qam == 64 else 8
        uses = (nb * n + m - 1) // m
        assert resources["blocks"] == nb and resources["source_padding_bits"] == nb * k - bits
        assert resources["payload_uses"] == uses and resources["QAM_padding_bits"] == uses * m - nb * n
        assert resources["total_uses"] == uses and resources["total_energy"] == uses
    files[p.name] = {"sha256": sha(p), "bytes": len(raw), "native_shape": list(array.shape),
                     "cross_build_decoded_pixels_identical": backend_agreement}

for record in server["smoke"]["channels"]:
    p = artifacts / Path(record["arrays_path"]).name
    assert sha(p) == record["arrays_file_sha256"]
    with np.load(p, allow_pickle=False) as archive:
        assert set(archive.files) == {"transmitted", "received"}
        x, out = archive["transmitted"], archive["received"]
    assert array_sha(x) == record["transmitted"]["sha256"]
    assert array_sha(out) == record["received"]["sha256"]
    noise_rng, fade_rng = np.random.default_rng(), np.random.default_rng()
    noise_rng.bit_generator.state = record["rng_before"]["noise"]
    fade_rng.bit_generator.state = record["rng_before"]["fading"]
    noise, h = np.zeros(len(x), dtype=complex), np.ones(len(x), dtype=complex)
    if record["channel"] != "identity":
        draws = noise_rng.standard_normal(x.shape) * math.sqrt(10 ** (-record["snr_db"] / 10) / 2)
        noise = draws[:, 0] + 1j * draws[:, 1]
        if record["channel"] == "rayleigh":
            draws = fade_rng.standard_normal(x.shape) / math.sqrt(2)
            h = draws[:, 0] + 1j * draws[:, 1]
    xc = x[:, 0] + 1j * x[:, 1]
    decoded = (h * xc + noise) / h
    expected = np.column_stack((decoded.real, decoded.imag)).astype(x.dtype)
    # Linux/macOS complex division can differ in the final FP64 rounding bit.
    # Serialized arrays and raw RNG draws remain SHA-exact; report arithmetic
    # replay error separately rather than pretending cross-build bit equality.
    maximum_error = float(np.max(np.abs(out - expected)))
    tolerance = 64 * np.finfo(x.dtype).eps * max(1.0, float(np.max(np.abs(out))))
    np.testing.assert_allclose(out, expected, rtol=0, atol=tolerance)
    assert array_sha(noise) == record["noise"]["sha256"] and array_sha(h) == record["fading"]["sha256"]
    assert noise_rng.bit_generator.state == record["rng_after"]["noise"]
    assert fade_rng.bit_generator.state == record["rng_after"]["fading"]
    assert record["uses"] == len(x) == 4096 and record["pilot_uses"] == 0
    assert math.isclose(float(np.sum(x ** 2)), record["energy"], rel_tol=1e-14)
    files[p.name] = {"sha256": sha(p), "RNG_draws_SHA_exact": True,
                     "cross_build_replay_max_absolute_error": maximum_error,
                     "cross_build_replay_absolute_tolerance": tolerance, "uses": len(x)}

report = {"state": "passed", "scope": "local_and_sheng_CPU_engineering_ONLY",
          "server_evidence_sha256": sha(server_path), "protocol_sha256": server["protocol_sha256"],
          "source_files": server["source_files"], "artifacts": files,
          "native_frame": "000000", "GT_or_main_validation_read": False,
          "LDPC_simulation_completed": False, "AP_completed": False}
with OUT.open("x") as handle:
    json.dump(report, handle, indent=2)
print(json.dumps({"state": "passed", "artifacts": len(files), "output": str(OUT)}))
