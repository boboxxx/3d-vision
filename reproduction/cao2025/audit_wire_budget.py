"""Independently derive native air-slot costs from previously audited ROI records.

This is a deterministic layout audit, not a trained transmission or AP run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--roi-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('preserve earlier evidence; choose a unique output')
    audit = json.loads(args.roi_audit.read_text())
    assert audit['state'] == 'passed' and audit['records_sha256'] == digest(args.records)
    rows = [json.loads(line) for line in args.records.read_text().splitlines()]
    assert len(rows) == audit['frames'] and len({row['frame_id'] for row in rows}) == len(rows)
    values = []
    for row in rows:
        h, w = row['views'][0]['shape']
        assert all(view['shape'] == [h, w] for view in row['views'])
        hp, wp = h+(-h)%6, w+(-w)%6
        cells = []
        for view in row['views']:
            occupied = np.zeros((hp//2, wp//2), dtype=np.bool_)
            for box in view['boxes']:
                x1, y1, x2, y2 = box['xyxy']
                assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
                occupied[y1//2:(y2+1)//2, x1//2:(x2+1)//2] = True
            cells.append(int(occupied.sum()))
        real = 9*(2*(hp//6)*(wp//6)+sum(cells))
        data = (real+1)//2
        boxes = sum(len(view['boxes']) for view in row['views'])
        # Separately serialized13-byte prefix and12-byte boxes, each with CRC32.
        control = (13+4+boxes*12+4)*8*7
        rgb = 2*3*h*w
        values.append(dict(frame_id=row['frame_id'], shape=[h, w], key_cells=cells,
            data_real_values=real, odd_padding_real_values=real%2, data_uses=data,
            control_uses=control, awgn_total_uses=data+control,
            rayleigh_total_uses=data+control+8, rgb_real_values=rgb))
    summary = {}
    for key in ('data_uses', 'control_uses', 'awgn_total_uses', 'rayleigh_total_uses'):
        numbers = [value[key] for value in values]
        summary[key] = dict(min=min(numbers), max=max(numbers), mean=float(np.mean(numbers)), total=sum(numbers))
    rgb_total = sum(value['rgb_real_values'] for value in values)
    summary['aggregate_awgn_CBR_complex_per_rgb_real_value'] = summary['awgn_total_uses']['total']/rgb_total
    summary['aggregate_rayleigh_CBR_complex_per_rgb_real_value'] = summary['rayleigh_total_uses']['total']/rgb_total
    summary['control_fraction_of_awgn_uses'] = summary['control_uses']['total']/summary['awgn_total_uses']['total']
    frame_path = args.output.with_suffix('.frames.jsonl')
    if frame_path.exists():
        parser.error('preserve earlier frame accounting')
    frame_path.write_text(''.join(json.dumps(value)+'\n' for value in values))
    root = Path(__file__).parent
    result = dict(state='passed', scope='independent fixed-layout cost audit; no trained channel/RGB/AP claim',
        frames=len(values), split=audit['split'], summary=summary,
        records_sha256=digest(args.records), inherited_roi_audit_sha256=digest(args.roi_audit),
        per_frame_sha256=digest(frame_path), source_sha256=digest(__file__),
        implemented_protocol_sources={name: digest(root/name) for name in ('radio.py', 'wireless.py')},
        limitations='image identity inherited from ROI audit; sparse transmission uses dense CNN computation; '
                    'unit energy is enforced by transmitter, not newly measured on trained frames; '
                    'analog data have no asserted digital bitrate/compression ratio')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
