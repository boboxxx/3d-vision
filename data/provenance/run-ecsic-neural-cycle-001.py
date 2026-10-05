"""One pinned sheng CPU bridge and independent raw-array audit; no GPU/new PHY."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'experiments/original-paper-scenarios/ecsic-neural-reception-code'
sys.path.insert(0, str(HERE))
import contract as c

PREFIX = 'ecsic-neural-reception-native-001'


def main():
    output = ROOT / 'data/runs' / (PREFIX + '-cycle.json')
    directory = Path('/mnt/d/paper6/runs') / PREFIX
    audit = ROOT / 'data/provenance' / (PREFIX + '-audit.json')
    c.need(not output.exists() and not directory.exists() and not audit.exists(), 'unique native paths')
    c.need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('WANDB_MODE') == 'disabled', 'CPU-only bridge controller')
    sources = c.identities(); _, predecessor = c.predecessor()
    for name in ('ecsic-neural-local-CPU-003.json', 'ecsic-neural-sheng-CPU-001.json'):
        gate = c.read(ROOT / 'data/engineering' / name)
        c.need(gate['state'] == 'passed' and gate['tests'] == 5 and gate['sources'] == sources, 'same current five CPU gates')
    info = dict(state='running', prefix=PREFIX, pid=os.getpid(), started_unix=time.time(),
                sources=sources, predecessor=predecessor, completed_commands=[], no_new_channel_draws=True,
                no_GPU_or_KITTI_AP=True, output_dir=str(directory))
    c.save(output, info)
    try:
        for label, command in [('neural-crop', [HERE / 'run.py', '--directory', directory]),
                               ('independent-audit', [HERE / 'audit.py', '--directory', directory, '--output', audit])]:
            c.need(c.identities() == sources, 'fixed receiver sources changed')
            log = ROOT / 'logs' / (PREFIX + '-' + label + '.log')
            with log.open('x') as stream:
                process = subprocess.Popen([sys.executable, *map(str, command)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                returncode = process.wait()
            info['completed_commands'].append(dict(label=label, pid=process.pid, returncode=returncode,
                log_path=str(log.relative_to(ROOT)), log_sha256=c.sha(log)))
            c.save(output, info); c.need(returncode == 0, 'native CPU command failed: ' + label)
            c.need(c.identities() == sources and c.predecessor()[1] == predecessor, 'sources/predecessors changed')
        reviewed = c.read(audit)
        c.need(reviewed['state'] == 'passed_all20_outcomes11_causal_neural_and_exact_crops' and
               reviewed['full_arrays_exact'] == 198 and reviewed['cropped_views_exact'] == 22, 'complete raw neural audit')
        info.update(state='finished_all20_outcomes11_neural_crop_pairs_and_audit_pending_terminal_closure',
                    ended_unix=time.time(), audit_sha256=c.sha(audit), manifest_sha256=c.sha(directory / 'manifest.json'))
    except BaseException as error:
        info.update(state='failed', exception=repr(error), ended_unix=time.time()); raise
    finally: c.save(output, info)
    print(json.dumps(dict(state=info['state'], received=11, erased=9)), flush=True)


if __name__ == '__main__': main()
