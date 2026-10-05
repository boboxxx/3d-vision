"""Prelocked four-arm3340-update/16-final-endpoint cycle with full independent audits."""
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
from integration import ROOT, source_specs
from common import check, sha256, source_identity, evaluation_sources
from conditions import PARENT_SHA, PROTOCOL_SHA, CONFIG_SHA, paired_records
from eligibility import verify_eligibility

ENGINEERING = 'stereo-epipolar-native-sanity-001'
INTEGRATION = 'stereo-epipolar-evaluation-integration-001'
ARMS = ('U', 'G', 'P', 'S')
ENDPOINTS = (('identity', 10), ('awgn', 6), ('awgn', 10), ('awgn', 18))


def read(path): return json.loads(Path(path).read_text())


def training_engineering_gate(closure, local, closure_sha):
    check(closure['actual_terminal'] and closure['actual_training_records'] == 24,
          'complete four-arm native engineering')
    check(local['state'] == 'passed_all_transferred_engineering_artifacts_and24_raw_rows' and
          local['closure_sha256'] == closure_sha and local['artifacts_verified'] == 27 and
          local['records'] == 24 and local['paired_steps'] == 6,
          'exact complete local training engineering closure')
    return dict(state='passed', closure_sha256=closure_sha, records=24, artifacts=27)


def integration_gate(identities):
    closure_path = ROOT / 'data/provenance' / (INTEGRATION + '-closure.json')
    local_path = ROOT / 'data/provenance' / (INTEGRATION + '-local-verification-001.json')
    closure, local = read(closure_path), read(local_path)
    check(closure['actual_terminal'] and closure['frames'] == 48 and closure['references'] == 8 and
          local['state'] == 'passed' and local['actual_prediction_frames'] == 48 and
          local['exact_receiver_references'] == 8 and local['closure_sha256'] == sha256(closure_path), 'complete native/local integration')
    check(local['artifacts'] == len(closure['artifacts_sha256']) == 96, 'all96 artifacts locally verified')
    check(all(sha256(ROOT / p) == h for p, h in closure['artifacts_sha256'].items()), 'complete integration artifacts changed')
    actual_ps = subprocess.run(['ps', '-p', str(closure['cycle_pid']), '-o', 'pid,stat,args'], capture_output=True, text=True)
    check(actual_ps.returncode == 1 and len(actual_ps.stdout.splitlines()) <= 1, 'prior integration PID actually absent')
    saved = read(ROOT / 'data/provenance' / ('server-source-manifest-' + INTEGRATION + '.json'))
    check(all(identities[k] == saved[k]['actual_sources'] for k in identities if k != 'evaluation'), 'same immutable seven native trees')
    old = saved['evaluation']['actual_sources']['file_hashes']
    current = identities['evaluation']['file_hashes']
    check(all(current.get(p) == h for p, h in old.items() if p != 'experiments/geometry-link/F9/evaluation/check_cpu.py'),
          'all measured native evaluation producers/configs unchanged')
    return dict(closure_sha256=sha256(closure_path), local_verification_sha256=sha256(local_path), actual_PID_absent=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    check(re.fullmatch(r'[a-z0-9][a-z0-9-]{2,100}', args.prefix), 'safe unique prefix')
    os.chdir(ROOT); runs = ROOT / 'data/runs'; logs = ROOT / 'logs'
    output = runs / (args.prefix + '-cycle.json')
    check(not output.exists(), 'preserve prior cycle')
    specs = source_specs(); identities = {k: source_identity(*v) for k, v in specs.items()}
    eligibility = verify_eligibility(); integration = integration_gate(identities)
    engineer_path = ROOT / 'data/provenance' / (ENGINEERING + '-closure.json')
    engineer = read(engineer_path)
    local_engineer = read(ROOT / 'data/provenance' / (ENGINEERING + '-local-verification-001.json'))
    engineering_gate = training_engineering_gate(engineer, local_engineer, sha256(engineer_path))
    check(all(sha256(ROOT / p) == h for p, h in engineer['artifacts_sha256'].items()), 'all27 original engineering artifacts unchanged')
    parent = Path('/mnt/d/paper6/runs/stereo-encoder-native-seed17-001-joint/checkpoint_epoch_1.pth')
    protocol = ROOT / 'experiments/geometry-link/F9-epipolar-conditioned-JSCC.md'
    config = ROOT / 'configs/tuning/stereo_task_seed17_epoch1.yaml'; fold = ROOT / 'data/internal-tuning-fold-001.json'
    check(sha256(parent) == PARENT_SHA and sha256(protocol) == PROTOCOL_SHA and sha256(config) == CONFIG_SHA, 'sole parent/prelocked protocol/config')
    gates = [ROOT / 'data/engineering/F9-evaluation-local-CPU-007.json', ROOT / 'data/engineering/F9-evaluation-sheng-CPU-002.json']
    for path in gates:
        cpu = read(path)
        check(cpu['state'] == 'passed' and cpu['tests'] == 12 and cpu['evaluation_source_identity'] == evaluation_sources(), 'current twelve CPU final-interface gates')
    source_path = ROOT / 'data/provenance' / ('server-source-manifest-' + args.prefix + '.json')
    eval_path = ROOT / 'data/provenance' / ('server-evaluation-source-manifest-' + args.prefix + '.json')
    check(not source_path.exists() and not eval_path.exists(), 'unique formal manifests')
    for arm in ARMS:
        check(not (runs / f'{args.prefix}-{arm}.json').exists() and
              not (Path('/mnt/d/paper6/runs') / f'{args.prefix}-{arm}').exists(), 'unique training outputs')
        for channel, snr in ENDPOINTS:
            rid = f'{args.prefix}-{arm}-test-{channel}{snr}'
            check(all(not (runs / (rid + suffix)).exists() for suffix in
                      ('.json', '-features.jsonl', '-boundary-report.json', '-AP-audit.json', '-feature-audit.json')), 'unique final endpoints')
    source_path.write_text(json.dumps({k: dict(actual_sources=v) for k, v in identities.items() if k != 'evaluation'}, indent=2) + '\n')
    eval_path.write_text(json.dumps(dict(evaluation=identities['evaluation'],
                                       native_seven_tree_manifest_sha256=sha256(source_path)), indent=2) + '\n')
    protocol_paths = [protocol, ROOT / 'experiments/geometry-link/F9/evaluation-integration-protocol-001.md',
                      ROOT / 'experiments/geometry-link/F9/receiver-path-clarification-001.md',
                      ROOT / 'experiments/geometry-link/F9/formal-cycle-interface-001.md']
    info = dict(state='starting', prefix=args.prefix, pid=os.getpid(), started_unix=time.time(),
                engineering=False, updates_per_arm=3340, planned_endpoints=16, source_freeze_active=True,
                source_manifest_sha256=sha256(source_path), evaluation_manifest_sha256=sha256(eval_path),
                source_identities=identities, integration=integration, eligibility=eligibility, training_engineering=engineering_gate,
                protocols_sha256={str(p.relative_to(ROOT)): sha256(p) for p in protocol_paths},
                CPU_gates_sha256={str(p.relative_to(ROOT)): sha256(p) for p in gates},
                completed_commands=[], training={}, evaluations={})
    def save(): output.write_text(json.dumps(info, indent=2, allow_nan=False) + '\n')
    def frozen():
        check(identities == {k: source_identity(*v) for k, v in specs.items()}, 'formal eight-tree source freeze')
        check(all(sha256(ROOT / p) == h for p, h in {**info['protocols_sha256'], **info['CPU_gates_sha256']}.items()), 'locked evidence changed')
        check(sha256(source_path) == info['source_manifest_sha256'] and sha256(eval_path) == info['evaluation_manifest_sha256'], 'formal source manifests changed')
    def invoke(label, command):
        frozen(); path = logs / (args.prefix + '-' + label + '.log')
        with path.open('x') as stream:
            result = subprocess.run([sys.executable, *map(str, command)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        info['completed_commands'].append(dict(label=label, log_path=str(path.relative_to(ROOT)), log_sha256=sha256(path), returncode=result.returncode))
        save(); check(result.returncode == 0, 'formal command failed: ' + label); frozen()
    def memory_gate():
        while True:
            frozen()
            values = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], text=True).strip().splitlines()
            check(len(values) == 1, 'single GPU'); used, total = (int(v.strip()) for v in values[0].split(','))
            if (total-used) * 2**20 >= 12 * 2**30: break
            info.update(state='waiting_physical_GPU_margin', last_free_MiB=total-used); save(); time.sleep(15)
        info['state'] = 'running'; save()
    save()
    try:
        common = ['--config', config, '--protocol', protocol, '--fold', fold, '--source-manifest', source_path]
        core = ROOT / 'experiments/geometry-link/F9/code'
        for arm in ARMS:
            memory_gate(); info.update(current_stage='training', current_arm=arm); save()
            rid = args.prefix + '-' + arm; manifest = runs / (rid + '.json'); audit = runs / (rid + '-training-audit.json')
            invoke(arm + '-train', [core / 'train.py', *common, '--checkpoint', parent, '--arm', arm,
                                   '--output-dir', Path('/mnt/d/paper6/runs') / rid, '--manifest', manifest])
            invoke(arm + '-training-audit', [core / 'audit.py', *common, '--initialization', parent, '--manifest', manifest, '--output', audit])
            value, audited = read(manifest), read(audit)
            check(value['optimizer_steps'] == audited['steps'] == 3340 and audited['state'] == 'passed', 'whole final training audit')
            info['training'][arm] = dict(manifest_path=str(manifest.relative_to(ROOT)), manifest_sha256=sha256(manifest),
                                        audit_path=str(audit.relative_to(ROOT)), audit_sha256=sha256(audit)); save()
        paired = paired_records({a: Path(read(ROOT / v['manifest_path'])['output_dir']) / 'training.jsonl' for a, v in info['training'].items()}, 3340)
        pair_path = runs / (args.prefix + '-pair-audit.json')
        with pair_path.open('x') as stream: json.dump(paired, stream, indent=2)
        check(paired['state'] == 'passed', 'full four-arm actual input/noise pairing')
        info['paired_audit_sha256'] = sha256(pair_path); frozen(); save()
        for arm in ARMS:
            train = read(ROOT / info['training'][arm]['manifest_path'])
            checkpoint = Path(train['checkpoint_path']); checkpoint_sha = train['checkpoint_sha256']
            for channel, snr in ENDPOINTS:
                memory_gate(); info.update(current_stage='final_evaluation', current_arm=arm, current_channel=channel, current_snr=snr); save()
                rid = f'{args.prefix}-{arm}-test-{channel}{snr}'; label = f'{arm}-{channel}{snr}'
                manifest = runs / (rid + '.json'); features = runs / (rid + '-features.jsonl'); boundary = runs / (rid + '-boundary-report.json')
                ap = runs / (rid + '-AP-audit.json'); feature = runs / (rid + '-feature-audit.json')
                invoke(label + '-test', ['-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=1',
                    HERE / 'diagnose.py', '--records', features, '--report', boundary, '--checkpoint-sha256', checkpoint_sha,
                    '--arm', arm, '--expected-channel', channel, '--expected-snr', snr,
                    'test', '--seed', '17', '--manifest', manifest, '--protocol', protocol, '--cfg_file', HERE / 'configs' / f'{channel}{snr}-holdout.yaml',
                    '--ckpt', checkpoint, '--batch_size', '1', '--workers', '4', '--launcher', 'pytorch', '--save_to_file', '--eval_tag', rid])
                invoke(label + '-AP-audit', [HERE / 'AP_audit.py', '--manifest', manifest, '--source-manifest', source_path,
                    '--preparation', ROOT / 'data/kitti-preparation-001.json', '--tuning-fold', fold, '--stereo-feature-link',
                    *(['--identity-codec-diagnostic'] if channel == 'identity' else []), '--arm', arm, '--expected-channel', channel,
                    '--expected-snr', snr, '--boundary-report', boundary, '--checkpoint-sha256', checkpoint_sha, '--output', ap])
                invoke(label + '-feature-audit', [HERE / 'feature_audit.py', '--manifest', manifest, '--features', features, '--report', boundary,
                    '--AP-audit', ap, '--fold', fold, '--expected-channel', channel, '--expected-snr', snr, '--arm', arm,
                    '--checkpoint-sha256', checkpoint_sha, '--output', feature])
                run, br, aa, fa = (read(p) for p in (manifest, boundary, ap, feature))
                check(run['state'] == br['state'] == 'finished' and aa['state'] == fa['state'] == 'passed', 'complete independent endpoint audits')
                check(aa['frames'] == fa['frames'] == br['frames'] == 372 and fa['readonly_states'] == 539, 'full372/539 endpoint')
                check(aa['run_manifest_sha256'] == fa['run_manifest_sha256'] == sha256(manifest) and
                      fa['AP_audit_sha256'] == sha256(ap) and fa['boundary_report_sha256'] == aa['boundary_report_sha256'] == sha256(boundary) and
                      fa['feature_records_sha256'] == sha256(features), 'complete endpoint identity')
                check(run['run_checkpoint_sha256'] == br['checkpoint_sha256'] == fa['checkpoint_sha256'] == checkpoint_sha, 'sole same-arm final checkpoint')
                check(br['evaluation_sources'] == aa['evaluation_sources'] == fa['evaluation_sources'] == identities['evaluation'], 'complete eval source identity')
                info['evaluations'][label] = dict(arm=arm, channel=channel, snr_db=snr, checkpoint_sha256=checkpoint_sha,
                    artifacts_sha256={str(p.relative_to(ROOT)): sha256(p) for p in (manifest, boundary, features, ap, feature)},
                    Car3D_AP_R40_percent={k: v for k, v in aa['recomputed_metrics'].items() if k.startswith('Car_3d/')}); save()
        check(len(info['evaluations']) == 16, 'all16 endpoints before differences')
        def moderate(arm): return info['evaluations'][arm + '-awgn10']['Car3D_AP_R40_percent']['Car_3d/moderate_R40']
        info['prelocked_AWGN10_Moderate_differences_pp'] = {f'P_minus_{a}': moderate('P') - moderate(a) for a in ('G', 'S', 'U')}
        frozen(); info.update(state='finished_all_four_training_pair_16_AP_feature_audits_pending_terminal_closure', ended_unix=time.time(),
            source_freeze_active=False, current_stage=None, current_arm=None,
            limitations='Single internal fold/seed with author detector pretraining exposure; geometry effect interpretation awaits terminal/local closure; no mainval/fading/multiseed/project completion'); save()
        print(json.dumps(dict(state=info['state'], endpoints=16)), flush=True)
    except BaseException as error:
        info.update(state='failed', exception=repr(error), ended_unix=time.time()); save(); raise


if __name__ == '__main__': main()
