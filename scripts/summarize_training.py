#!/usr/bin/env python3
"""Read CRC-checked tensorboardX scalar evidence; no training modification."""
import argparse
import json
import math
from pathlib import Path
import statistics
import struct
import sys
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from geocomm.evidence import sha256


def record_payloads(path, crc):
    """An active writer may end with one incomplete record, never bad CRC."""
    with path.open('rb') as stream:
        while True:
            header = stream.read(8)
            if len(header) < 8:
                return
            header_crc = stream.read(4)
            if len(header_crc) < 4:
                return
            if struct.unpack('<I', header_crc)[0] != crc(header):
                raise ValueError('event header CRC mismatch')
            length = struct.unpack('<Q', header)[0]
            if length > 64 * 1024 * 1024:
                raise ValueError('unexpected event size')
            payload = stream.read(length)
            payload_crc = stream.read(4)
            if len(payload) != length or len(payload_crc) != 4:
                return
            if struct.unpack('<I', payload_crc)[0] != crc(payload):
                raise ValueError('event payload CRC mismatch')
            yield payload


def scalar_summary(rows):
    values = {}
    for step, value in rows:
        if not math.isfinite(value):
            raise ValueError('nonfinite saved scalar')
        if step in values and values[step] != value:
            raise ValueError('conflicting duplicate scalar step')
        values[step] = value
    steps = sorted(values)
    ordered = [values[x] for x in steps]
    if not ordered:
        raise ValueError('empty scalar')
    window = min(100, len(steps))
    return dict(unique_steps=len(steps), first_step=steps[0], last_step=steps[-1],
                min=min(ordered), max=max(ordered),
                first_window_median=statistics.median(ordered[:window]),
                last_window_median=statistics.median(ordered[-window:]),
                window_steps=window), values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--events', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-steps', type=int)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('preserve prior snapshots; choose unique output')
    from tensorboardX.proto.event_pb2 import Event
    from tensorboardX.record_writer import masked_crc32c
    paths = sorted(args.events.glob('events.out.tfevents.*'))
    if not paths:
        raise ValueError('no event files')
    snapshots = args.output.with_suffix('.events')
    snapshots.mkdir(parents=True, exist_ok=False)
    for path in paths:
        shutil.copyfile(path, snapshots/path.name)
    paths = sorted(snapshots.iterdir())
    rows = {}
    for path in paths:
        for payload in record_payloads(path, masked_crc32c):
            event = Event.FromString(payload)
            for value in event.summary.value:
                if value.HasField('simple_value'):
                    rows.setdefault(value.tag, []).append((event.step, value.simple_value))
    summaries, by_step = {}, {}
    for tag, values in rows.items():
        summaries[tag], by_step[tag] = scalar_summary(values)
    required = ['train/loss', 'train/gradient_norm_before_clip',
                'train/communication_complex_uses', 'train/communication_energy']
    if not all(tag in summaries for tag in required):
        raise ValueError('required training evidence missing')
    complete = set.intersection(*(set(by_step[tag]) for tag in required))
    if not complete or complete != set(range(1, max(complete) + 1)):
        raise ValueError('training evidence has noncontiguous complete steps')
    if args.expected_steps and complete != set(range(1, args.expected_steps + 1)):
        raise ValueError('completed epoch does not match locked step budget')
    for step in complete:
        uses = by_step[required[2]][step]
        energy = by_step[required[3]][step]
        if uses <= 0 or abs(energy / uses - 1) > 1e-4:
            raise ValueError('channel symbol/energy mismatch')
    run = json.loads(args.manifest.read_text())
    if run['mode'] != 'train' or not run['codec_only']:
        raise ValueError('not a codec-only training run')
    result = dict(state='passed', evidence_type='saved_training_scalars_only',
                  training_run_state=run['state'], completed_evidence_steps=max(complete),
                  manifest_sha256=sha256(args.manifest),
                  event_files={str(p): dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in paths},
                  scalars=summaries,
                  limitations='finite loss/preclip norm and optimizer-loop step records; checkpoint updates and AP require separate audit; source files can grow after these immutable event snapshots')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(steps=max(complete), loss=summaries['train/loss'],
                         preclip=summaries['train/gradient_norm_before_clip'])))


if __name__ == '__main__':
    main()
