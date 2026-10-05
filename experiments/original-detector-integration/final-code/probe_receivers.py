"""One-frame real full768 engineering for new cache/receiver execution, no AP."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from common import ROOT, HERE, check, current_sources, read, save, sha256, tensor_identity, terminal
from receivers import LigaReceiver, StereoReceiver
from geocomm.pooling_diagnostic import state_hashes
sys.path[:0] = [str(HERE.parent), str(ROOT / 'reproduction/cao2025')]
from adapter import transmit_rgb
from data import StereoRGB
from wireless import WirelessVariant
import radio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector', choices=('liga', 'stereo_rcnn'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    check(not args.output.exists(), 'retain previous native engineering')
    queue = read(ROOT / 'data/runs/native-validation-queue-002.json')
    check(queue['state'] == 'running' and queue['current_command'] == 'final-' + args.detector
          and queue['pid'] == os.getppid() and not terminal(queue['pid']), 'actual queue002 native child scope')
    f7_closure_path = ROOT / 'data/provenance/stereo-channel-native-sanity-002-closure.json'
    f7_closure = read(f7_closure_path)
    f7_cycle = read(ROOT / 'data/runs/stereo-channel-native-sanity-002-cycle.json')
    check(sha256(f7_closure_path) == queue['F7_engineering_closure_sha256']
          and f7_closure['state'] == 'closed_all_audits_passed' and f7_closure['engineering']
          and f7_closure['training_updates_per_arm'] == 6 and terminal(f7_cycle['pid']),
          'repaired native predecessor terminal closure')
    free = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                                    '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    check(len(free) == 1 and int(free[0]) >= 12288, 'physical singleGPU12GiB margin')
    free_bytes = int(free[0]) * 2**20
    torch.set_num_threads(2); torch.manual_seed(17); torch.cuda.manual_seed_all(17)
    sources = current_sources()
    record = dict(state='starting', detector=args.detector, scope='one-frame first000036 fresh768 engineering only; no AP',
                  started_at_unix=time.time(), source_identities=sources, conditions=[], NVIDIA_free_before_bytes=free_bytes)
    save(args.output, record)
    receiver = None
    try:
        receiver = (LigaReceiver if args.detector == 'liga' else StereoReceiver)()
        spec = importlib.util.spec_from_file_location('paper6_final_engineering_cao', ROOT / 'reproduction/cao2025/model.py')
        semantic = importlib.util.module_from_spec(spec); spec.loader.exec_module(semantic)
        codec = WirelessVariant(semantic.SemanticVariant('/mnt/d/paper6/checkpoints/spynet_sintel_final-3d2a1287.pth')).cpu().eval()
        initial = Path('/mnt/d/paper6/runs/cao2025-full-native-seed17-001-stage1/initialization.pth')
        check(sha256(initial) == 'f4d0d5d5bcc203ee5fd0554a0e691b2a530c56cb55d48fe21b6fc0f97331d3cb', 'fresh untrained init')
        state = torch.load(initial, map_location='cpu', weights_only=False)['model_state']
        check(len(state) == len(codec.state_dict()) == 768, 'full768 init')
        codec.load_state_dict(state, strict=True); del state
        before = state_hashes(codec)

        def forbidden(*args, **kwargs): raise RuntimeError('broad training/clean MSE forbidden')
        codec.forward = forbidden; codec.semantic_mse = forbidden
        dataset = StereoRGB('/mnt/d/paper6/data/kitti', ROOT / 'data/engineering/cao2025-roi-holdout-001.jsonl',
                            ROOT / 'data/engineering/cao2025-roi-holdout-audit-001.json',
                            ROOT / 'data/internal-tuning-fold-001.json', 'geocomm_tune_holdout')
        frame = dataset[0]; check(frame['frame_id'] == '000036', 'fixed native first frame')
        directory = Path('/mnt/d/paper6/runs') / args.output.stem
        check(not directory.exists(), 'retain engineering files'); directory.mkdir(parents=True)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for channel in ('clean_relay', 'identity', 'awgn'):
                result = dict(outputs=(frame['left'], frame['right']), erasure=None, accounting=None) if channel == 'clean_relay' else \
                    transmit_rgb(codec, radio, frame['left'], frame['right'], frame['boxes'], channel, 10., torch.Generator().manual_seed(17))
                images = result['outputs']
                tensor_hashes = [tensor_identity(v) for v in images] if images is not None else None
                tensor_path = directory / f'{channel}-received.pth'
                torch.save(dict(frame_id='000036', channel=channel, outputs=images), tensor_path)
                received = torch.load(tensor_path, map_location='cpu', weights_only=False)
                check(set(received) == {'frame_id', 'channel', 'outputs'} and received['frame_id'] == '000036'
                      and received['channel'] == channel, 'lossless envelope')
                images = received['outputs']
                check(([tensor_identity(v) for v in images] if images is not None else None) == tensor_hashes, 'lossless received cache bytes')
                predictions = directory / channel; predictions.mkdir()
                if images is None:
                    (predictions / '000036.txt').write_text(''); endpoint = dict(prediction_count=0, calls={})
                else:
                    endpoint = receiver.predict(images, '000036', Path('/mnt/d/paper6/data/kitti/training/calib/000036.txt'), predictions)
                torch.cuda.synchronize()
                lines = (predictions / '000036.txt').read_text().splitlines()
                check(len(lines) == endpoint['prediction_count'] and all(len(line.split()) == 16 and
                      all(__import__('math').isfinite(float(v)) for v in line.split()[1:]) for line in lines),
                      'complete finite native writer')
                check(state_hashes(receiver.model) == receiver.initial and state_hashes(codec) == before, 'full detector/codec readonly')
                check(torch.cuda.max_memory_reserved() + 2 * 2**30 <= free_bytes, 'measured physical memory margin')
                record['conditions'].append(dict(channel=channel, received_tensors=tensor_hashes, accounting=result['accounting'],
                                                  erasure=result['erasure'], received_file_sha256=sha256(tensor_path), **endpoint))
                save(args.output, record)
                print(json.dumps(dict(detector=args.detector, channel=channel, count=endpoint['prediction_count'])), flush=True)
        check(current_sources() == sources, 'native engineering source closure')
        record.update(state='passed', detector_states_readonly=receiver.states, codec_states_readonly=768,
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(), ended_at_unix=time.time(),
                      limitations='Fresh untrained codec, one frame, no AP/fullfold/latency/trained baseline claim')
        save(args.output, record)
    except BaseException as exc:
        record.update(state='failed', exception=repr(exc), ended_at_unix=time.time()); save(args.output, record); raise
    finally:
        if receiver is not None: receiver.close()


if __name__ == '__main__': main()
