"""Fresh engineering queue002; no overlap with original train and no formal AP."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
F7 = ROOT / 'experiments/geometry-link/F7/code'
FINAL = ROOT / 'experiments/original-detector-integration/final-code'
sys.path[:0] = [str(ROOT / 'src'), str(F7), str(FINAL)]
from geocomm.evidence import sha256, source_identity
from train import source_specs
from common import current_sources, terminal, NATIVE_PROBES, check


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--identity', type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    identity = read(args.identity)
    output = ROOT / 'data/runs/native-validation-queue-002.json'
    check(not output.exists(), 'unique queue002 required')
    failed = ROOT / 'data/provenance/F7-first-native-engineering-failure-001-closure.json'
    check(sha256(failed) == identity['failed_engineering_closure_sha256'] and
          read(failed)['state'] == 'closed_zero_optimizer_updates_no_learned_checkpoint', 'retain zero-update failure')

    def frozen():
        check({k: source_identity(*v)['sha256'] for k, v in source_specs().items()} == identity['F7_sources_sha256'], 'F7 source changed')
        check({k: v['sha256'] for k, v in current_sources().items()} == identity['final_sources_sha256'], 'final source changed')
        check(sha256(Path(__file__)) == identity['queue_sha256'], 'queue source changed')
        check(sha256(ROOT / identity['CPU_evidence_path']) == identity['CPU_evidence_sha256'], 'CPU evidence changed')
        check(read(ROOT / identity['CPU_evidence_path'])['state'] == 'passed', 'repaired CPU checks required')
        check(sha256(ROOT / identity['repair_addendum_path']) == identity['repair_addendum_sha256'], 'repair addendum changed')
        original = read(ROOT / 'data/provenance/cao2025-staged-trainer-source-bfee9d8.json')
        check(len(original) == 21 and all(sha256(ROOT / 'reproduction/cao2025' / k) == v for k, v in original.items()), 'original21 changed')

    frozen()
    result = dict(state='waiting_original_cycle_terminal_and_stage5_audit', pid=os.getpid(),
                  identity_sha256=sha256(args.identity), started_at_unix=time.time(), completed_commands=[],
                  scope='repaired F7 six-step pair and final receiver GPU engineering only; no formal AP')
    save(output, result)

    def invoke(label, command):
        frozen()
        log = ROOT / 'logs' / ('native-validation-queue-002-' + label + '.log')
        result.update(state='running', current_command=label)
        save(output, result)
        with log.open('x') as stream:
            subprocess.run([sys.executable, *map(str, command)], cwd=ROOT,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        result['completed_commands'].append(dict(label=label, log_sha256=sha256(log)))
        frozen()
        save(output, result)

    try:
        while True:
            stage_path = ROOT / 'data/runs/cao2025-full-native-seed17-001-stage5.json'
            audit_path = stage_path.with_name(stage_path.stem + '-audit.json')
            stage = read(stage_path)
            check(stage['state'] != 'failed', 'original stage5 failed; preserve and stop')
            if terminal(26642) and stage['state'] == 'finished' and audit_path.exists():
                audit = read(audit_path)
                check(audit['state'] == 'passed' and audit['manifest_sha256'] == sha256(stage_path), 'stage5 audit must pass')
                check(len(stage['epochs']) == len(audit['epochs']) == 45 and stage['samples'] == audit['samples'] == 150300, 'full stage5 budget')
                check(sha256(stage['epochs'][-1]['checkpoint_path']) == audit['checkpoint_sha256'], 'final original checkpoint identity')
                result['original_stage5_audit_sha256'] = sha256(audit_path)
                break
            result.update(last_original_state=stage['state'], last_checked_at_unix=time.time(),
                          original_cycle_terminal=terminal(26642))
            save(output, result)
            time.sleep(15)
        while True:
            memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip().splitlines()
            check(len(memory) == 1, 'one physical GPU')
            if int(memory[0]) >= 12288:
                break
            result.update(state='waiting_physical_GPU_margin', last_free_MiB=int(memory[0]))
            save(output, result)
            time.sleep(15)
        prefix = 'stereo-channel-native-sanity-002'
        invoke('F7-cycle', [F7 / 'cycle.py', '--prefix', prefix, '--engineering'])
        invoke('F7-close', [F7 / 'close.py', '--prefix', prefix])
        closure_path = ROOT / 'data/provenance' / (prefix + '-closure.json')
        closure = read(closure_path)
        check(closure['state'] == 'closed_all_audits_passed' and closure['engineering'] and
              closure['training_updates_per_arm'] == 6, 'repaired paired engineering must close')
        result['F7_engineering_closure_sha256'] = sha256(closure_path)
        for detector, name in NATIVE_PROBES.items():
            probe_path = ROOT / 'data/engineering' / name
            check(not probe_path.exists(), 'preserve any previous final probe')
            invoke('final-' + detector, [FINAL / 'probe_receivers.py', '--detector', detector, '--output', probe_path])
            probe = read(probe_path)
            check(probe['state'] == 'passed' and probe['codec_states_readonly'] == 768 and
                  probe['source_identities'] == current_sources(), 'new real final-receiver source/read-only evidence')
        a, b = [read(ROOT / 'data/engineering' / NATIVE_PROBES[d]) for d in ('stereo_rcnn', 'liga')]
        check(len(a['conditions']) == len(b['conditions']) == 3, 'all three native receiver conditions')
        for left, right in zip(a['conditions'], b['conditions']):
            check(all(left[k] == right[k] for k in ('channel', 'received_tensors', 'accounting', 'erasure')), 'paired received native inputs')
        result.update(state='finished_all_native_engineering_closed', ended_at_unix=time.time(),
                      native_probe_sha256={k: sha256(ROOT / 'data/engineering' / v) for k, v in NATIVE_PROBES.items()},
                      formal_training_or_AP_started=False)
        save(output, result)
    except BaseException as error:
        result.update(state='failed', exception=repr(error), ended_at_unix=time.time())
        save(output, result)
        raise


if __name__ == '__main__':
    main()
