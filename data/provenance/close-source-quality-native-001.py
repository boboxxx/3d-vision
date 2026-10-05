"""Actual terminal quality process and complete native three-file closure."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/sheng/paper6')
SCRIPT = ROOT / 'data/provenance/native-source-quality-001.py'
spec = importlib.util.spec_from_file_location('quality_closed', SCRIPT)
q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--scope', choices=('engineering', 'main'), required=True)
    args = parser.parse_args(); prefix = 'source-quality-' + args.scope + '-001'
    closure = ROOT / 'data/provenance' / (prefix + '-closure.json'); assert not closure.exists()
    source, ids, _, _, _, dependencies = q.gate(args.scope)
    manifest = ROOT / 'data/runs' / (prefix + '.json'); audit = ROOT / 'data/provenance' / (prefix + '-audit.json')
    run, report = q.r.read(manifest), q.r.read(audit)
    assert run['state'] == 'finished_all_six_native_quality_conditions'
    assert report['state'] == 'passed_every_native_quality_record_fresh_pixels_integer_MSE_and_frozen_SSIM_replay'
    process = subprocess.run(['ps', '-p', str(run['pid']), '-o', 'pid,stat,args'], capture_output=True, text=True)
    assert process.returncode == 1
    # Audit is a separate process and must also have exited; don't rely on
    # inability to read protected /proc/io for detecting actual processes.
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():
            continue
        try:
            argv = path.joinpath('cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        assert not any(arg in (os.fsencode(str(SCRIPT)), b'data/provenance/native-source-quality-001.py') for arg in argv)
    assert run['sources'] == report['sources'] == source and run['dependencies'] == report['dependencies'] == dependencies
    assert report['manifest_sha256'] == q.r.sha(manifest) and run['records_sha256'] == report['records_sha256']
    assert run['views_done'] == report['views'] == 12 * len(ids) and run['conditions'] == report['conditions']
    raw = Path(run['records_path']); assert q.r.sha(raw) == run['records_sha256']
    paths = (manifest, audit, raw)
    def mapped(path):
        if path.is_relative_to(ROOT):
            return str(path.relative_to(ROOT))
        assert path.is_relative_to('/mnt/d/paper6/runs/' + prefix)
        return 'data/engineering/' + prefix + '-native-transfer/' + str(path).lstrip('/')
    result = dict(state='closed_native_quality_actual_terminal_all_records', actual_terminal=True,
                  scope=args.scope, checked_unix=time.time(), pid=run['pid'], sources=source, dependencies=dependencies,
                  frames_each_condition=len(ids), views=report['views'], conditions=report['conditions'],
                  artifacts_sha256={mapped(path): q.r.sha(path) for path in paths},
                  native_to_local_artifacts={mapped(path): str(path) for path in paths},
                  closure_script_sha256=q.r.sha(__file__))
    with closure.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(state=result['state'], views=result['views'], artifacts=3)))


if __name__ == '__main__':
    main()
