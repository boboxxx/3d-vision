"""Bounded transfer of closed ROI/quality metadata; no raw image/cache copies."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HOST = 'sheng@100.94.183.27'
REMOTE = '/home/sheng/paper6'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o',
       'ControlPath=/private/tmp/paper6-sheng-01a10237.sock']
PREFIXES = [base + scope + '-001' for base in ('public-val-ROI-', 'source-quality-')
            for scope in ('engineering', 'main')]


def run(command):
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prefix', choices=PREFIXES, required=True)
    args = parser.parse_args(); prefix = args.prefix
    closure = ROOT / 'data/provenance' / (prefix + '-closure.json'); assert not closure.exists()
    run(['rsync', '-az', '-e', shlex.join(SSH), HOST + ':' + REMOTE + '/data/provenance/' + closure.name,
         str(closure.parent) + '/'])
    record = json.loads(closure.read_text()); assert record['actual_terminal']
    count = 8 if prefix.startswith('public-val-ROI-') else 3
    assert len(record['native_to_local_artifacts']) == len(record['artifacts_sha256']) == count
    roots, native = [], []
    for local, path in record['native_to_local_artifacts'].items():
        value = Path(path); assert value.is_absolute() and '..' not in value.parts
        if value.is_relative_to(REMOTE):
            relative = value.relative_to(REMOTE)
            assert relative.parts[:2] in (('data', 'runs'), ('data', 'provenance')) or relative.parts[0] == 'logs'
            assert relative.name.startswith(prefix) and local == str(relative)
            roots.append(str(relative))
        else:
            assert value.is_relative_to('/mnt/d/paper6/runs/' + prefix)
            assert value.name == ('records.jsonl' if prefix.startswith('public-val-ROI-') else 'quality.jsonl')
            expected = 'data/engineering/' + prefix + '-native-transfer/' + path.lstrip('/')
            assert local == expected; native.append(path.lstrip('/'))
    assert len(native) == 1 and len(roots) == count - 1
    root_list = ROOT / 'data/provenance' / (prefix + '-transfer-files.txt')
    native_list = ROOT / 'data/provenance' / (prefix + '-native-transfer-files.txt')
    for path, values in ((root_list, roots), (native_list, native)):
        with path.open('x') as stream:
            stream.write('\n'.join(sorted(values)) + '\n')
    mirror = ROOT / 'data/engineering' / (prefix + '-native-transfer'); mirror.mkdir()
    run(['rsync', '-az', '--files-from=' + str(root_list), '-e', shlex.join(SSH), HOST + ':' + REMOTE + '/', str(ROOT) + '/'])
    run(['rsync', '-az', '--files-from=' + str(native_list), '-e', shlex.join(SSH), HOST + ':/', str(mirror) + '/'])
    print(json.dumps(dict(state='transferred_all_closed_native_metadata_artifacts', prefix=prefix, artifacts=count)))


if __name__ == '__main__':
    main()
