"""Independent complete training-byte records/selection/source terminal audit."""
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
PREFIX = 'original-jpeg-rate-calibration-001'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def main():
    output = ROOT / 'data/provenance' / (PREFIX + '-audit.json')
    assert not output.exists()
    manifest_path = ROOT / 'data/runs' / (PREFIX + '.json')
    info = read(manifest_path)
    assert info['state'] == 'finished6080_real_pair_encodes_independent_audit_pending'
    ps = subprocess.run(['ps', '-p', str(info['pid']), '-o', 'pid,stat,etime,args'], capture_output=True, text=True)
    assert ps.returncode == 1 and len(ps.stdout.splitlines()) <= 1
    assert info['actual_records'] == 6080 and info['completed_qualities'] == 95
    fold = read(ROOT / 'data/internal-tuning-fold-001.json')
    ids = sorted(fold['folds']['geocomm_tune_train']['ids'])[:64]
    assert info['frame_ids'] == ids and len(ids) == 64 and set(ids).isdisjoint(fold['folds']['geocomm_tune_holdout']['ids'])
    for p, digest in [(ROOT / 'experiments/original-paper-scenarios/jpeg-rate-calibration-protocol-001.md', info['protocol_sha256']),
                      (ROOT / 'data/internal-tuning-fold-001.json', info['fold_sha256']),
                      (ROOT / 'data/engineering/original-scenarios-sheng-CPU-001.json', info['CPU_gate_sha256'])]: assert sha(p) == digest
    for group in info['sources'].values():
        for p, digest in group.items(): assert sha(ROOT / p) == digest
    expected_paths = {str(Path('/mnt/d/paper6/data/kitti/training') / view / (frame + '.png')) for frame in ids for view in ('image_2', 'image_3')}
    assert set(info['source_images_before']) == expected_paths and info['source_images_after'] == info['source_images_before']
    for p, digest in info['source_images_before'].items(): assert sha(p) == digest
    raw = Path(info['records_path']); assert sha(raw) == info['records_sha256']
    totals = {q: [0, 0] for q in range(1, 96)}; inputs = {}; received_shapes = {}; rows = 0
    with raw.open() as stream:
        for index, line in enumerate(stream):
            row = json.loads(line); quality = index // 64 + 1; frame = ids[index % 64]
            assert quality <= 95 and row['frame_id'] == frame and row['quality'] == row['parameter'] == quality
            assert row['codec'] == 'jpeg' and row['options'] == dict(quality=quality, subsampling=2, optimize=False, progressive=False)
            assert row['framing_bytes'] == 20 and row['source_bytes'] == 20 + sum(row['codestream_bytes']) and row['source_bits'] == 8 * row['source_bytes']
            assert len(row['codestream_bytes']) == 2 and all(type(x) is int and x > 0 for x in row['codestream_bytes'])
            assert re.fullmatch('[a-f0-9]{64}', row['wire_sha256'])
            pair = (row['left_input'], row['right_input'])
            if frame in inputs: assert pair == inputs[frame]
            inputs[frame] = pair
            assert pair[0]['shape'] == pair[1]['shape'] and len(pair[0]['shape']) == 3 and pair[0]['shape'][-1] == 3
            count = math.prod(pair[0]['shape'])
            assert row['raw_RGB8_bits'] == count * 16
            assert row['actual_raw_to_serialized_ratio'] == count * 2 / row['source_bytes']
            assert row['left_received']['shape'] == row['right_received']['shape'] == pair[0]['shape']
            totals[quality][0] += count * 2; totals[quality][1] += row['source_bytes']; rows += 1
    assert rows == 6080
    ratios = {str(q): dict(raw_RGB8_bytes=r, full_wire_bytes=w, ratio=r / w) for q, (r, w) in totals.items()}
    assert ratios == info['pooled_ratios']
    selected = {}
    for target in (10, 30, 50):
        ordered = sorted((abs(math.log(totals[q][0] / totals[q][1] / target)), q) for q in totals)
        residual, quality = ordered[0]; ratio = totals[quality][0] / totals[quality][1]
        selected[str(target)] = dict(quality=quality, measured_ratio=ratio, absolute_log_residual=residual)
    assert selected == info['selected_qualities'] and info['no_GT_model_mainval_or_radio'] and info['input_barrier'] and not info['candidate_wires_retained']
    result = dict(state='passed6080_records128_source_files_three_prelocked_rate_choices_actual_terminal', checked_unix=time.time(),
        manifest_sha256=sha(manifest_path), records_sha256=sha(raw), rows=6080, source_files=128, actual_terminal=True,
        actual_ps=ps.stdout, ps_returncode=ps.returncode, pid=info['pid'], selected_qualities=selected,
        sources=info['sources'], protocol_sha256=info['protocol_sha256'], candidate_wires_reencoded=False,
        scope='Independent complete byte-accounting/selection audit; actual encoding covered by sealed native encoder, no wire re-encoding or AP')
    with output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(state=result['state'], selected=selected)), flush=True)


if __name__ == '__main__': main()
