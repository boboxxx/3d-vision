"""Repaired F7/final receiver regressions, exclusive evidence and frozen sources."""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'experiments/geometry-link/F7/code'),
               str(ROOT / 'experiments/original-detector-integration/final-code')]
from geocomm.evidence import sha256, source_identity
from train import source_specs
from common import current_sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    log = args.output.with_suffix('.log')
    if args.output.exists() or log.exists():
        raise RuntimeError('unique evidence ID required')
    f7 = {k: source_identity(*v) for k, v in source_specs().items()}
    final = current_sources()
    commands = [
        'experiments/geometry-link/F7/code/test_conditions.py',
        'experiments/geometry-link/F7/code/test_native_scope.py',
        'tests/test_stereo_task_adaptation.py',
        'experiments/original-detector-integration/final-code/test_contracts.py']
    record = dict(state='running', scope='repaired_CPU_contracts_only_no_native_GPU_AP',
                  started_at_unix=time.time(), sources_F7=f7, sources_final=final,
                  script_sha256=sha256(Path(__file__)),
                  test_source_sha256={name: sha256(ROOT / name) for name in commands})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(record, stream, indent=2)
    try:
        tests = 0
        with log.open('x') as stream:
            for name in commands:
                result = subprocess.run([sys.executable, str(ROOT / name), '-v'], cwd=ROOT,
                    env=dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'), capture_output=True, text=True)
                stream.write(json.dumps(dict(command=name, returncode=result.returncode)) + '\n' + result.stdout + result.stderr)
                stream.flush()
                if result.returncode != 0:
                    raise RuntimeError('regression failed: ' + name)
                match = re.search(r'Ran (\d+) tests?', result.stderr)
                if match is None:
                    raise RuntimeError('test count unavailable')
                tests += int(match.group(1))
        if tests != 19:
            raise RuntimeError('unexpected regression count')
        if f7 != {k: source_identity(*v) for k, v in source_specs().items()} or final != current_sources():
            raise RuntimeError('source changed during checks')
        original = json.loads((ROOT / 'data/provenance/cao2025-staged-trainer-source-bfee9d8.json').read_text())
        if len(original) != 21 or not all(sha256(ROOT / 'reproduction/cao2025' / k) == v for k, v in original.items()):
            raise RuntimeError('original21 changed')
        record.update(state='passed', tests=19, F7_tests=14, final_tests=5, original21_unchanged=True)
    except Exception as error:
        record.update(state='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        record.update(finished_at_unix=time.time(), log_sha256=sha256(log))
        args.output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(state='passed', tests=19, output=str(args.output))))


if __name__ == '__main__':
    main()
