"""Require full F8 closure and terminal evidence before native F9 work."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify_eligibility():
    closure_path = ROOT / 'data/provenance/stereo-encoder-native-seed17-001-closure.json'
    closure = read(closure_path)
    assert closure['state'] == 'closed_all_audits_passed' and not closure['engineering']
    assert closure['training_updates_per_arm'] == 3340 and len(closure['evaluations']) == 4
    for name, expected in closure['artifacts_sha256'].items():
        assert sha(ROOT / name) == expected, name
    local_path = ROOT / 'data/provenance/stereo-encoder-native-seed17-001-local-verification-001.json'
    local = read(local_path)
    assert local['state'] == 'passed' and local['closure_sha256'] == sha(closure_path)
    assert local['paired_steps'] == 3340 and local['artifacts_verified'] == 46
    terminal_path = ROOT / 'data/provenance/stereo-encoder-native-seed17-001-actual-terminal-001.json'
    terminal = read(terminal_path)
    assert terminal['cycle_actual_terminal'] and terminal['ps_returncode'] == 1
    assert terminal['closure_sha256'] == sha(closure_path)
    assert terminal['artifacts_sha256'] == closure['artifacts_sha256']
    current = subprocess.run(['ps', '-p', str(terminal['cycle_pid']), '-o', 'args='], text=True, capture_output=True)
    assert current.returncode == 1 and not current.stdout.strip(), 'preceding PID exists; inspect actual process before proceeding'
    return dict(state='passed_full_F8_closure_actual_terminal',
                artifacts_sha256={str(p.relative_to(ROOT)): sha(p) for p in (closure_path, local_path, terminal_path)},
                actual_terminal_PID=terminal['cycle_pid'])
