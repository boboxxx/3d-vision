"""CPU-only quadrature vs independent complex-Gaussian channel verification."""
import argparse
import csv
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import erfc, erfcx

PI = math.pi
SPAN = 57.6
N = 1048576
SEED = 2027100509
ANGLES = [-PI / 2, -PI / 3, 0.0, PI / 3, PI / 2]
FILES = [
    "data/provenance/check-position-phase-theory-009.py",
    "experiments/cost-field/position-phase-theory-protocol-009.md",
    "experiments/cost-field/position-phase-derivation-009.md",
    "experiments/cost-field/code_004/codec.py",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pdf(phi, q, kind):
    a = math.cos(phi)
    if kind == "awgn":
        base = math.exp(-1 / q) / (2 * PI)
        if a <= 0:
            return base * (1 + a * math.sqrt(PI / q) * erfcx(-a / math.sqrt(q)))
        return base + a / (2 * math.sqrt(PI * q)) * math.exp(-math.sin(phi)**2 / q) * erfc(-a / math.sqrt(q))
    b = math.sqrt(q + math.sin(phi)**2)
    return q / (2 * PI * b*b) * (1 + a / b * math.atan2(b, -a))


def integrate(fn, theta=0.0):
    cuts = {-PI, 0.0, PI}
    for edge in [-PI, -PI/2, PI/2, PI]:
        for cycle in [-1, 0, 1]:
            v = edge - theta + 2*PI*cycle
            if -PI < v < PI:
                cuts.add(v)
    values = sorted(cuts)
    total = err = 0.0
    for left, right in zip(values[:-1], values[1:]):
        val, e = quad(fn, left, right, epsabs=2e-11, epsrel=2e-11, limit=300)
        total += val
        err += e
    return total, err


def wrapped(t):
    return (t + PI) % (2*PI) - PI


def moments(theta, q, kind):
    def loss(phi):
        received = max(-PI/2, min(PI/2, wrapped(theta + phi)))
        return ((received - theta)*SPAN/PI)**2 * pdf(phi, q, kind)
    def clipped(phi):
        return float(abs(wrapped(theta + phi)) > PI/2) * pdf(phi, q, kind)
    mse, me = integrate(loss, theta)
    prob, pe = integrate(clipped, theta)
    return mse, prob, me, pe


def run(root, out):
    out.mkdir(parents=True, exist_ok=False)
    report = {"state": "running", "started_unix": time.time(),
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
              "hostname": platform.node(), "source_sha256": {f: sha(root/f) for f in FILES},
              "draws_per_condition": N, "seed": SEED, "rows": [], "checks": []}
    path = out / "report.json"
    rng = np.random.Generator(np.random.PCG64(SEED))
    try:
        for kind in ["awgn", "rayleigh"]:
            for snr in range(6, 19):
                q = 10**(-snr/10)
                grid = np.linspace(-PI, PI, 4097)
                vals = np.array([pdf(float(p), q, kind) for p in grid])
                mass, mass_err = integrate(lambda p: pdf(p, q, kind))
                sym = float(np.max(np.abs(vals - vals[::-1])))
                assert vals.min() >= 0 and abs(mass - 1) <= 2e-10 and sym <= 2e-12
                radials = []
                for phi in np.linspace(-PI, PI, 13):
                    a = math.cos(float(phi))
                    if kind == "awgn":
                        fn = lambda r: r / (PI*q) * math.exp(-(r*r+1-2*r*a)/q)
                    else:
                        fn = lambda r: q/PI*r/(q+r*r+1-2*r*a)**2
                    ref, re = quad(fn, 0, np.inf, epsabs=2e-11, epsrel=2e-11, limit=300)
                    closed = pdf(float(phi), q, kind)
                    delta = abs(closed-ref)
                    assert delta <= 2e-10+2e-9*abs(ref)
                    radials.append({"phi": float(phi), "pdf": closed, "radial": ref, "radial_error": re, "difference": delta})
                noise = rng.standard_normal((N, 2))*math.sqrt(q/2)
                n = noise[:, 0] + 1j*noise[:, 1]
                del noise
                h = None
                if kind == "rayleigh":
                    fading = rng.standard_normal((N, 2))/math.sqrt(2)
                    h = fading[:, 0] + 1j*fading[:, 1]
                    assert np.all(np.abs(h) > 0)
                    del fading
                theory = []
                for theta in ANGLES:
                    mse, prob, me, pe = moments(theta, q, kind)
                    theory.append((mse, prob))
                    assert 0 <= mse <= SPAN**2 and 0 <= prob <= 1
                    x = complex(math.cos(theta), math.sin(theta))
                    # Independent physical channel and atan2 decoder; does not sample pdf().
                    rx = x+n if h is None else (h*x+n)/h
                    phase = np.angle(rx)
                    small = np.maximum(np.abs(rx.real), np.abs(rx.imag)) <= np.finfo(np.float32).eps
                    phase[small] = 0
                    indicator = np.abs(phase) > PI/2
                    err2 = ((np.clip(phase, -PI/2, PI/2)-theta)*SPAN/PI)**2
                    observed = float(err2.mean())
                    se = float(err2.std(ddof=1)/math.sqrt(N))
                    observed_prob = float(indicator.mean())
                    prob_se = math.sqrt(prob*(1-prob)/N)
                    row = {"channel": kind, "snr_db": snr, "theta": theta, "theta_over_pi": theta/PI,
                           "exact_mse_m2": mse, "exact_rmse_m": math.sqrt(mse), "exact_clipping_probability": prob,
                           "mc_mse_m2": observed, "mc_rmse_m": math.sqrt(observed), "mc_mse_SE_m2": se,
                           "mc_clipping_probability": observed_prob, "clipping_SE": prob_se,
                           "mse_z": abs(observed-mse)/se, "quadrature_mse_error": me, "quadrature_probability_error": pe,
                           "fallback_events": int(small.sum()),
                           "fallback_probability_bound": 4*np.finfo(np.float32).eps**2/(PI*q),
                           "passed": abs(observed-mse) <= 8*se+1e-8 and abs(observed_prob-prob) <= 8*prob_se+8/N}
                    report["rows"].append(row)
                    assert row["passed"], row
                assert max(abs(theory[i][j]-theory[-i-1][j]) for i in [0,1] for j in [0,1]) <= 2e-9
                report["checks"].append({"channel": kind, "snr_db": snr, "mass": mass, "mass_error_bound": mass_err,
                                         "min_pdf": float(vals.min()), "pdf_symmetry_error": sym, "radial_checks": radials})
                path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
                print(json.dumps({"channel": kind, "snr_db": snr, "rows_completed": len(report["rows"])}), flush=True)
        assert len(report["rows"]) == 130 and len(report["checks"]) == 26
        report["state"] = "passed_all130_exact_phase_moments_and_direct_Gaussian_checks"
        report["draws_total_physical_channels"] = N*26
        report["coordinate_observations"] = N*130
        report["maximum_mse_standard_errors"] = max(r["mse_z"] for r in report["rows"])
        report["maximum_fallback_mse_difference_bound_m2"] = max(r["fallback_probability_bound"]*SPAN**2 for r in report["rows"])
        report["no_AP_or_detector_error_claim"] = True
        with (out/"moments.csv").open("x", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(report["rows"][0]))
            writer.writeheader()
            writer.writerows(report["rows"])
    except Exception as exc:
        report["state"] = "failed_retained"
        report["failure"] = repr(exc)
        raise
    finally:
        report["ended_unix"] = time.time()
        report["source_sha256_after"] = {f: sha(root/f) for f in FILES}
        assert report["source_sha256"] == report["source_sha256_after"]
        path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
