"""Unique eight-condition smoke plus independent audit, no training or AP."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from integration import ROOT, source_specs, IDS
from common import check, sha256, source_identity, evaluation_sources


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    check(re.fullmatch(r'[a-z0-9][a-z0-9-]{2,100}', args.prefix), 'safe prefix')
    os.chdir(ROOT)
    report = ROOT / 'data/runs' / (args.prefix + '-cycle.json')
    source_path = ROOT / 'data/provenance' / ('server-source-manifest-' + args.prefix + '.json')
    check(not report.exists() and not source_path.exists(), 'preserve prior cycle/source manifest')
    cpu_path = ROOT / 'data/engineering/F9-evaluation-sheng-CPU-001.json'
    cpu = json.loads(cpu_path.read_text())
    check(cpu['state'] == 'passed' and cpu['tests'] == 10 and cpu['evaluation_source_identity'] == evaluation_sources(), 'current ten CPU gates required')
    local_path = ROOT / 'data/engineering/F9-evaluation-local-CPU-005.json'
    local = json.loads(local_path.read_text())
    check(local['state'] == 'passed' and local['tests'] == 10 and local['evaluation_source_identity'] == evaluation_sources(), 'matched current local CPU gates')
    specs = source_specs()
    identities = {k: source_identity(*v) for k, v in specs.items()}
    sealed = json.loads((ROOT / 'data/provenance/server-source-manifest-stereo-epipolar-native-sanity-001.json').read_text())
    check(set(sealed) == set(identities) - {'evaluation'}, 'same immutable seven-tree identities')
    check(all(identities[k] == sealed[k]['actual_sources'] for k in sealed), 'sealed native sources changed')
    for arm in ('U', 'G', 'P', 'S'):
        for channel in ('identity', 'awgn'):
            rid = f'{args.prefix}-{arm}-{channel}10'
            check(not (ROOT / 'data/runs' / (rid + '.json')).exists() and
                  not (Path('/mnt/d/paper6/runs') / rid).exists(), 'all eight unique output paths')
    source_path.write_text(json.dumps({k: dict(actual_sources=v) for k, v in identities.items()}, indent=2) + '\n')
    protocols = [ROOT / 'experiments/geometry-link/F9/evaluation-integration-protocol-001.md',
                 ROOT / 'experiments/geometry-link/F9/receiver-path-clarification-001.md']
    info = dict(state='starting', pid=os.getpid(), prefix=args.prefix, engineering_only=True, no_AP=True,
                started_unix=time.time(), source_manifest_sha256=sha256(source_path),
                protocols_sha256={str(p.relative_to(ROOT)): sha256(p) for p in protocols},
                CPU_gates_sha256={str(p.relative_to(ROOT)): sha256(p) for p in (cpu_path, local_path)},
                ids=IDS, conditions={}, completed_commands=[])
    def save(): report.write_text(json.dumps(info, indent=2, allow_nan=False) + '\n')
    def frozen():
        check(identities == {k: source_identity(*v) for k, v in specs.items()}, 'eight-tree source freeze')
        check(all(sha256(ROOT / p) == h for p, h in info['protocols_sha256'].items()), 'prelocked protocol changed')
    def invoke(label, command):
        frozen()
        log = ROOT / 'logs' / (args.prefix + '-' + label + '.log')
        with log.open('x') as stream:
            run = subprocess.run([sys.executable, *map(str, command)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        info['completed_commands'].append(dict(label=label, returncode=run.returncode,
            log_path=str(log.relative_to(ROOT)), log_sha256=sha256(log)))
        save(); check(run.returncode == 0, 'native command failed: ' + label); frozen()
    save()
    try:
        for arm in ('U', 'G', 'P', 'S'):
            for channel in ('identity', 'awgn'):
                label = f'{arm}-{channel}10'; rid = args.prefix + '-' + label
                manifest = ROOT / 'data/runs' / (rid + '.json')
                info.update(state='running', current_condition=label); save()
                invoke(label, [HERE / 'integration.py', '--arm', arm, '--channel', channel,
                    '--output-dir', Path('/mnt/d/paper6/runs') / rid, '--report', manifest, '--source-manifest', source_path])
                value = json.loads(manifest.read_text())
                check(value['state'] == 'passed' and value['frames'] == 6, 'six native frames passed')
                info['conditions'][label] = dict(report_path=str(manifest.relative_to(ROOT)), report_sha256=sha256(manifest)); save()
        audit = ROOT / 'data/runs' / (args.prefix + '-audit.json')
        invoke('independent-audit', [HERE / 'integration_audit.py', '--cycle', report, '--output', audit])
        info.update(state='finished_all48_native_frames_eight_references_and_audit_pending_terminal_closure',
                    current_condition=None, frames=48, references=8, audit_sha256=sha256(audit), ended_unix=time.time()); save()
    except BaseException as error:
        info.update(state='failed', exception=repr(error), ended_unix=time.time()); save(); raise
    print(json.dumps(dict(state=info['state'], frames=48, references=8)), flush=True)


if __name__ == '__main__': main()
