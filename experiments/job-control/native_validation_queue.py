"""Wait for physical GPU margin, then run only prelocked native engineering.

No job interruption, original training change, formal F7 launch or AP selection.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from geocomm.evidence import sha256, source_identity


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--identity', type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    identity = json.loads(args.identity.read_text())
    queue_path = ROOT / 'data/runs/native-validation-queue-001.json'
    probe_path = ROOT / 'data/engineering/original-StereoRCNN-RGB-GPU-endpoint-probe-001.json'
    prefix = 'stereo-channel-native-sanity-001'
    require(not queue_path.exists() and not probe_path.exists(), 'retain prior queue/probe')
    require(not (ROOT / f'data/runs/{prefix}-cycle.json').exists(), 'retain prior F7 cycle')
    specs = {
        'project': (ROOT, ['src', 'scripts', 'configs', 'pyproject.toml']),
        'liga': (ROOT / 'third_party/LIGA-Stereo', ['liga', 'configs', 'tools', 'setup.py']),
        'mmdet': (ROOT / 'third_party/mmdetection_kitti', ['mmdet']),
        'stereo_rcnn': (ROOT / 'third_party/Stereo-RCNN', ['lib', 'demo.py', 'test_net.py']),
        'experiment': (ROOT, ['experiments/geometry-link/F7/code']),
    }

    def frozen():
        for name, spec in specs.items():
            require(source_identity(*spec)['sha256'] == identity['sources_sha256'][name], 'changed source: ' + name)
        for name, value in identity['files_sha256'].items():
            require(sha256(ROOT / name) == value, 'changed dependency: ' + name)
        for name, value in identity['original21'].items():
            require(sha256(ROOT / 'reproduction/cao2025' / name) == value, 'changed original: ' + name)

    frozen()
    record = dict(state='starting', pid=os.getpid(), started_at_unix=time.time(),
                  identity_sha256=sha256(args.identity), completed_commands=[],
                  scope='native StereoRCNN engineering then paired F7 six-step engineering and terminal closure only')

    def save():
        temp = queue_path.with_suffix('.tmp')
        temp.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
        temp.replace(queue_path)

    def invoke(label, command):
        frozen()
        path = ROOT / 'logs' / (label + '.log')
        record.update(state='running', current_command=label)
        save()
        with path.open('x') as stream:
            subprocess.run([sys.executable, *map(str, command)], cwd=ROOT, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        frozen()
        record['completed_commands'].append(dict(label=label, log_sha256=sha256(path)))
        save()

    save()
    try:
        while True:
            lines = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                                             '--format=csv,noheader,nounits'], text=True).strip().splitlines()
            require(len(lines) == 1, 'single GPU required')
            free = int(lines[0])
            record.update(state='waiting_physical_GPU_margin', required_free_MiB=12288,
                          last_physical_free_MiB=free, last_checked_at_unix=time.time())
            save()
            if free >= 12288:
                break
            time.sleep(15)
        invoke('original-StereoRCNN-RGB-GPU-endpoint-probe-001', [
            ROOT / 'experiments/original-detector-integration/probe_stereo_rcnn_rgb_gpu.py',
            '--output', probe_path])
        probe = json.loads(probe_path.read_text())
        require(probe['state'] == 'passed' and probe['detector_states_readonly'] == 670
                and probe['codec_states_readonly'] == 768, 'native probe incomplete')
        record['terminal_probe_sha256'] = sha256(probe_path)
        # invoke waits/reaps the previous PID before starting the next process.
        invoke(prefix + '-cycle', [ROOT / 'experiments/geometry-link/F7/code/cycle.py',
                                  '--prefix', prefix, '--engineering'])
        invoke(prefix + '-close', [ROOT / 'experiments/geometry-link/F7/code/close.py',
                                  '--prefix', prefix])
        closure = ROOT / f'data/provenance/{prefix}-closure.json'
        require(json.loads(closure.read_text())['state'] == 'closed_all_audits_passed', 'F7 closure incomplete')
        record.update(state='finished_native_engineering_closed', ended_at_unix=time.time(),
                      F7_engineering_closure_sha256=sha256(closure), formal_F7_started=False)
        save()
    except BaseException as exc:
        record.update(state='failed', exception=repr(exc), ended_at_unix=time.time())
        save()
        raise


if __name__ == '__main__':
    main()
