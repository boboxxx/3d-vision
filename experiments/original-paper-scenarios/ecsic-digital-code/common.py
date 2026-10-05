"""Locked identities and fixed packet plan; no model or PHY import."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROTOCOL = ROOT / 'experiments/original-paper-scenarios/ecsic-digital-protocol-001.md'
PROTOCOL_SHA = 'da3c6a000af96628d954bbcdd2af8cce8343c3b7d66d731ceb9d5731af54208a'
STAGE_B = ROOT / 'data/engineering/ecsic-entropy-stageB-002'
AUDIT = ROOT / 'data/provenance/ecsic-entropy-stageB-002-audit.json'
AUDIT_SHA = '7fb7ca99b75054e7e099e2435a05ff94dff00fe1e8896ba320ed83d86bd99524'
MANIFEST_SHA = 'e098d9ec03e7783746ee9a150d0caf2b71e2c82816a47b088b3c4e6e54e6b867'
RUNTIME_SHA = '921c0cbb2b28d7671e62dfaf28b4ab504cd7956ebe9bdf53a2bb94762d61ecd1'
CDF_SHA = '507d22e35fb0bb605f05a3312600a8aff3ff10f1ec1c4964358cb390e35d56e5'
SOURCES = {
    'synthetic32x64': (813, 'acbc2be75dc60f74903dc19c57d100f2ea9772a9c4606f3fb7681e75a632a35f'),
    'training000000': (44824, 'e7242cca0a0edd43dbda26b8892894967a75690efeb1716ee538371c1905eb4a'),
}
CONFIGS = {'ldpc_2_3_qam64': (1296, 1944, 6), 'ldpc_1_2_qam256': (972, 1944, 8)}


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    # Caller reserves a unique output before calling; no previous run is reused.
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def identities():
    expected = read(HERE / 'historical-source-sha256.json')
    need(all(sha(ROOT / p) == value for p, value in expected.items()), 'historical source changed')
    need(sha(PROTOCOL) == PROTOCOL_SHA, 'locked protocol changed')
    current = {str(p.relative_to(ROOT)): sha(p) for p in sorted(HERE.iterdir()) if p.is_file()}
    return {**expected, **current, str(PROTOCOL.relative_to(ROOT)): PROTOCOL_SHA}


def source_payloads():
    need(sha(AUDIT) == AUDIT_SHA and sha(STAGE_B / 'manifest.json') == MANIFEST_SHA, 'stageB evidence identity')
    audit, manifest = read(AUDIT), read(STAGE_B / 'manifest.json')
    need(audit['state'] == 'passed' and audit['actual_native_terminal'], 'terminal stageB audit required')
    need(manifest['state'] == 'passed' and manifest['actual_entropy_source_bytes'], 'stageB bytes required')
    payloads = {}
    for name, (length, digest) in SOURCES.items():
        blob = (STAGE_B / name / 'source.p6ec').read_bytes()
        need(len(blob) == length and hashlib.sha256(blob).hexdigest() == digest, 'locked source bytes changed')
        need(audit['cases'][name]['payload_sha256'] == digest, 'source audit binding')
        payloads[name] = blob
    return payloads


def packet_plan():
    plan = []
    settings = [('identity', 10), ('awgn', 6), ('awgn', 18), ('rayleigh', 6), ('rayleigh', 18)]
    for source in SOURCES:
        for configuration in CONFIGS:
            for kind, snr in settings:
                j = len(plan)
                plan.append(dict(index=j, source=source, configuration=configuration, channel=kind, snr_db=snr,
                                 noise_seed=1911 + 2*j, fading_seed=1912 + 2*j,
                                 name=f'{j:02d}-{source}-{configuration}-{kind}-{snr}'))
    return plan
