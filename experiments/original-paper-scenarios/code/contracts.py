"""Original-inspired scenario contracts; no detector, GT, or training imports."""
import copy
import hashlib
import json
import math

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_array(value):
    value = np.ascontiguousarray(value)
    require(value.dtype.kind in "biufc" and np.isfinite(value).all(), "finite array required")
    descriptor = {"dtype": value.dtype.str, "shape": list(value.shape)}
    raw = json.dumps(descriptor, sort_keys=True).encode() + b"\0" + value.tobytes()
    return {**descriptor, "sha256": hashlib.sha256(raw).hexdigest()}


def real_symbols(value):
    value = np.asarray(value)
    require(value.dtype in (np.dtype("float32"), np.dtype("float64")), "FP32/64 required")
    require(value.ndim == 2 and value.shape[1] == 2 and len(value) > 0, "nonempty Nx2 I/Q")
    require(np.isfinite(value).all(), "nonfinite symbol")
    return value


def normalize_symbols(value):
    value = real_symbols(value)
    energy = np.mean(np.sum(value.astype(np.float64) ** 2, axis=1))
    require(energy > 0, "zero energy")
    return (value / math.sqrt(energy)).astype(value.dtype)


def zero_force(received, channel):
    """Exact ZF; deliberately no epsilon floor, MMSE or deep-fade clipping."""
    received, channel = np.asarray(received), np.asarray(channel)
    require(received.dtype.kind == channel.dtype.kind == "c", "complex arrays required")
    require(received.ndim == 1 and channel.shape == received.shape, "aligned channel shape")
    require(np.isfinite(received).all() and np.isfinite(channel).all(), "finite channel")
    require(np.all(np.abs(channel) > 0), "exact zero channel cannot be zero-forced")
    equalized = received / channel
    require(np.isfinite(equalized).all(), "nonfinite equalized signal")
    return equalized


def transmit(symbols, *, kind, snr_db, noise_rng, fading_rng):
    """Dedicated RNGs only. Main ambiguity-resolution variant uses iid fading."""
    symbols = real_symbols(symbols)
    require(kind in {"identity", "awgn", "rayleigh"}, "unknown channel")
    require(isinstance(noise_rng, np.random.Generator) and
            isinstance(fading_rng, np.random.Generator) and noise_rng is not fading_rng,
            "independent dedicated generators required")
    require(isinstance(noise_rng.bit_generator, np.random.PCG64) and
            isinstance(fading_rng.bit_generator, np.random.PCG64), "PCG64 required")
    require(math.isfinite(snr_db), "finite SNR required")
    power = float(np.mean(np.sum(symbols.astype(np.float64) ** 2, axis=1)))
    require(abs(power - 1.0) <= 1e-6, "normalize explicitly to Es=1 before transmission")
    before = {"noise": copy.deepcopy(noise_rng.bit_generator.state),
              "fading": copy.deepcopy(fading_rng.bit_generator.state)}
    x = symbols[:, 0].astype(np.float64) + 1j * symbols[:, 1].astype(np.float64)
    h = np.ones(len(x), dtype=np.complex128)
    noise = np.zeros(len(x), dtype=np.complex128)
    if kind != "identity":
        n0 = 10.0 ** (-snr_db / 10.0)
        require(math.isfinite(n0) and n0 > 0, "invalid noise power")
        draws = noise_rng.standard_normal((len(x), 2)) * math.sqrt(n0 / 2.0)
        noise = draws[:, 0] + 1j * draws[:, 1]
        if kind == "rayleigh":
            draws = fading_rng.standard_normal((len(x), 2)) / math.sqrt(2.0)
            h = draws[:, 0] + 1j * draws[:, 1]
    y = h * x + noise
    decoded = zero_force(y, h)
    output = np.column_stack((decoded.real, decoded.imag)).astype(symbols.dtype)
    require(np.isfinite(output).all(), "received dtype overflow")
    evidence = {
        "channel": kind, "snr_db": float(snr_db), "uses": len(x),
        "energy": float(power * len(x)), "Es": power,
        "pilot_uses": 0, "CSI": "perfect" if kind == "rayleigh" else "not_needed",
        "equalizer": "ZF" if kind == "rayleigh" else "identity",
        "coherence_symbols": 1 if kind == "rayleigh" else None,
        "noise": digest_array(noise), "fading": digest_array(h),
        "transmitted": digest_array(symbols), "received": digest_array(output),
        "rng_before": before,
        "rng_after": {"noise": copy.deepcopy(noise_rng.bit_generator.state),
                      "fading": copy.deepcopy(fading_rng.bit_generator.state)},
    }
    return output, evidence


def digital_resources(*, source_bits, n, k, qam, reliable_uses, reliable_energy, pilot_uses=0):
    """Block arithmetic ONLY; this does not encode, decode, or simulate LDPC."""
    values = (source_bits, n, k, qam, reliable_uses, pilot_uses)
    require(all(type(v) is int for v in values), "integer counts required")
    require(source_bits >= 0 and n > k > 0 and reliable_uses >= 0 and pilot_uses >= 0,
            "invalid resource count")
    require(qam in (64, 256), "reference QAM required")
    require((qam == 64 and 3 * k == 2 * n) or (qam == 256 and 2 * k == n),
            "reference LDPC rate mismatch")
    require(math.isfinite(reliable_energy) and reliable_energy >= 0,
            "finite nonnegative reliable-link energy")
    require(reliable_uses > 0 or reliable_energy == 0, "energy without reliable uses")
    bits_per_symbol = 6 if qam == 64 else 8
    blocks = (source_bits + k - 1) // k
    coded_bits = blocks * n
    payload_uses = (coded_bits + bits_per_symbol - 1) // bits_per_symbol
    return {
        "scope": "block_accounting_only_not_LDPC_simulation", "source_bits": source_bits,
        "n": n, "k": k, "QAM": qam, "blocks": blocks,
        "source_padding_bits": blocks * k - source_bits, "coded_bits": coded_bits,
        "QAM_padding_bits": payload_uses * bits_per_symbol - coded_bits,
        "payload_uses": payload_uses, "reliable_uses": reliable_uses,
        "pilot_uses": pilot_uses, "total_uses": payload_uses + reliable_uses + pilot_uses,
        "total_energy": payload_uses + float(reliable_energy) + pilot_uses,
        "nominal_source_bits_per_payload_use": 4.0,
    }


def scenarios():
    """One physical condition may feed multiple original-paper tables."""
    conditions = {}

    def add(table, source, cr, channel, snr, transport):
        key = f"{source}|cr{cr}|{channel}|snr{snr}|{transport}"
        if key not in conditions:
            conditions[key] = {"id": key, "source": source, "compression": cr,
                               "channel": channel, "snr_db": snr,
                               "transport": transport, "tables": []}
        conditions[key]["tables"].append(table)

    add("source", "clean", 1, "identity", None, "none")
    for cr in (10, 30, 50):
        for source in ("jpeg", "jpeg2000", "srcnn", "ecsic", "original_rgb"):
            add("source", source, cr, "identity", None, "source_only")
    for channel, grid in (("awgn", range(6, 19)), ("rayleigh", range(6, 19, 2))):
        for snr in grid:
            for transport in ("learned", "ldpc_2_3_qam64", "ldpc_1_2_qam256"):
                add("channel", "original_rgb", 30, channel, snr, transport)
    for snr in range(6, 19):
        add("joint", "original_rgb", 30, "awgn", snr, "learned")
        for source in ("jpeg", "jpeg2000", "ecsic"):
            for transport in ("ldpc_2_3_qam64", "ldpc_1_2_qam256"):
                add("joint", source, 30, "awgn", snr, transport)
    return list(conditions.values())
