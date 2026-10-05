"""Exactly six native updates per F9 arm, then full state/Adam/pair audits; no AP."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(HERE), str(ROOT / 'src')]
from geocomm.evidence import sha256, source_identity
from conditions import CONFIG_SHA, PARENT_SHA, PROTOCOL_SHA, check, paired_records
from eligibility import verify_eligibility
from train import source_specs


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    check(re.fullmatch(r'[a-z0-9][a-z0-9-]{2,100}', args.prefix) is not None, 'safe unique prefix')
    os.chdir(ROOT)
    runs, logs = ROOT / 'data/runs', ROOT / 'logs'
    output = runs / (args.prefix + '-cycle.json')
    check(not output.exists(), 'preserve prior cycle')
    parent = Path('/mnt/d/paper6/runs/stereo-encoder-native-seed17-001-joint/checkpoint_epoch_1.pth')
    protocol = ROOT / 'experiments/geometry-link/F9-epipolar-conditioned-JSCC.md'
    config = ROOT / 'configs/tuning/stereo_task_seed17_epoch1.yaml'
    fold = ROOT / 'data/internal-tuning-fold-001.json'
    check(sha256(parent) == PARENT_SHA and sha256(protocol) == PROTOCOL_SHA and sha256(config) == CONFIG_SHA, 'locked parent/protocol/config')
    eligibility = verify_eligibility()
    cpu_path = ROOT / 'data/engineering/F9-training-sheng-CPU-001.json'
    cpu = read(cpu_path)
    check(cpu['state'] == 'passed' and cpu['tests'] == 4 and cpu['CUDA_VISIBLE_DEVICES'] == '', 'four CPU training gates required')
    check(cpu['source_sha256'] == {str(p.relative_to(ROOT)): sha256(p) for p in HERE.glob('*.py')}, 'exact current CPU-checked implementation')
    core = read(ROOT / 'data/engineering/F9-coupling-sheng-CPU-001.json')
    check(core['state'] == 'passed' and core['tests_run'] == 11, 'independent dense core CPU gate required')
    check(all(sha256(ROOT / p) == h for p, h in core['source_sha256'].items()), 'sealed core gate')
    for arm in ('U', 'G', 'P', 'S'):
        check(not (runs / f'{args.prefix}-{arm}.json').exists() and not Path(f'/mnt/d/paper6/runs/{args.prefix}-{arm}').exists(), 'unique arm outputs')
    specs = source_specs()
    identities = {key: source_identity(*value) for key, value in specs.items()}
    old = read(ROOT / 'data/provenance/server-source-manifest-stereo-encoder-native-seed17-001.json')
    for key in ('project', 'liga', 'mmdet', 'stereo_rcnn', 'F7'):
        check(identities[key] == old[key]['actual_sources'], 'old source changed: ' + key)
    check(identities['F8'] == old['experiment']['actual_sources'], 'sealed F8 code changed')
    source_path = ROOT / 'data/provenance' / ('server-source-manifest-' + args.prefix + '.json')
    check(not source_path.exists(), 'unique source manifest')
    source_path.write_text(json.dumps({key: dict(git_revision=None, actual_sources=value) for key, value in identities.items()}, indent=2) + '\n')
    info = dict(state='starting', pid=os.getpid(), prefix=args.prefix, engineering=True, started_unix=time.time(),
                source_manifest_sha256=sha256(source_path), eligibility=eligibility, completed_commands=[], arms={},
                CPU_training_gate_sha256=sha256(cpu_path), native_AP_started=False)
    def save():
        output.write_text(json.dumps(info, indent=2, allow_nan=False) + '\n')
    def frozen():
        check(all(source_identity(*value) == identities[key] for key, value in specs.items()), 'native seven-tree source freeze changed')
    def invoke(label, command):
        frozen()
        log = logs / (args.prefix + '-' + label + '.log')
        with log.open('x') as stream:
            subprocess.run([sys.executable, *map(str, command)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        frozen()
        info['completed_commands'].append(dict(label=label, log_path=str(log.relative_to(ROOT)), log_sha256=sha256(log)))
        save()
    save()
    try:
        for arm in ('U', 'G', 'P', 'S'):
            while True:
                frozen()
                rows = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], text=True).strip().splitlines()
                check(len(rows) == 1, 'single GPU')
                used, total = (int(v.strip()) for v in rows[0].split(','))
                if (total-used)*2**20 >= 12*2**30: break
                info.update(state='waiting_physical_GPU_margin', current_arm=arm, last_physical_free_MiB=total-used); save(); time.sleep(15)
            info.update(state='running', current_arm=arm); save()
            rid = args.prefix + '-' + arm
            manifest, audit = runs / (rid + '.json'), runs / (rid + '-training-audit.json')
            common = ['--config', config, '--protocol', protocol, '--fold', fold, '--source-manifest', source_path]
            invoke(arm + '-train', [HERE / 'train.py', *common, '--checkpoint', parent, '--arm', arm, '--output-dir', Path('/mnt/d/paper6/runs') / rid, '--manifest', manifest, '--engineering-steps', '6'])
            invoke(arm + '-audit', [HERE / 'audit.py', *common, '--initialization', parent, '--manifest', manifest, '--output', audit, '--engineering-sanity'])
            info['arms'][arm] = dict(manifest_path=str(manifest), manifest_sha256=sha256(manifest), audit_path=str(audit), audit_sha256=sha256(audit)); save()
        pair = paired_records({arm: Path(read(item['manifest_path'])['output_dir']) / 'training.jsonl' for arm, item in info['arms'].items()}, 6)
        pair_path = runs / (args.prefix + '-pair-audit.json')
        with pair_path.open('x') as stream: json.dump(pair, stream, indent=2)
        frozen()
        info.update(state='finished_all_four_engineering_audits_pending_terminal_closure', current_arm=None,
                    ended_unix=time.time(), updates_per_arm=6, paired_audit_sha256=sha256(pair_path),
                    limitations='Six-step native engineering only; discarded weights, no AP/formal-method or project completion claim.')
        save()
    except BaseException as error:
        info.update(state='failed', exception=repr(error), ended_unix=time.time()); save(); raise
    print(json.dumps(dict(state=info['state'], updates_per_arm=6, arms=list(info['arms']))), flush=True)


if __name__ == '__main__':
    main()
