"""Independent complete transferred calibration-record verification; no PNG reads."""
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'original-jpeg-rate-calibration-001'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def main():
    output = ROOT / 'data/provenance' / (PREFIX + '-local-verification-001.json')
    assert not output.exists()
    manifest_path = ROOT / 'data/runs' / (PREFIX + '.json')
    audit_path = ROOT / 'data/provenance' / (PREFIX + '-audit.json')
    raw = ROOT / 'data/engineering' / PREFIX / 'calibration.jsonl'
    info, audit = read(manifest_path), read(audit_path)
    assert info['state'] == 'finished6080_real_pair_encodes_independent_audit_pending'
    assert audit['state'] == 'passed6080_records128_source_files_three_prelocked_rate_choices_actual_terminal'
    assert audit['actual_terminal'] and audit['ps_returncode'] == 1 and len(audit['actual_ps'].splitlines()) <= 1
    assert sha(manifest_path) == audit['manifest_sha256'] and sha(raw) == audit['records_sha256'] == info['records_sha256']
    for group in info['sources'].values():
        for p, digest in group.items(): assert sha(ROOT / p) == digest
    assert sha(ROOT / 'experiments/original-paper-scenarios/jpeg-rate-calibration-protocol-001.md') == info['protocol_sha256'] == audit['protocol_sha256']
    fold = read(ROOT / 'data/internal-tuning-fold-001.json')
    ids = sorted(fold['folds']['geocomm_tune_train']['ids'])[:64]
    assert ids == info['frame_ids'] and set(ids).isdisjoint(fold['folds']['geocomm_tune_holdout']['ids'])
    assert len(info['source_images_before']) == 128 and info['source_images_before'] == info['source_images_after']
    totals = {q: [0, 0] for q in range(1, 96)}; inputs = {}; rows = 0
    for index, line in enumerate(raw.read_text().splitlines()):
        row = json.loads(line); q = index // 64 + 1; frame = ids[index % 64]
        assert 1 <= q <= 95 and row['frame_id'] == frame and row['quality'] == row['parameter'] == q
        assert row['codec'] == 'jpeg' and row['options'] == dict(quality=q, subsampling=2, optimize=False, progressive=False)
        assert len(row['codestream_bytes']) == 2 and all(type(x) is int and x > 0 for x in row['codestream_bytes'])
        assert row['source_bytes'] == 20 + sum(row['codestream_bytes']) and row['framing_bytes'] == 20 and row['source_bits'] == 8 * row['source_bytes']
        pair = (row['left_input'], row['right_input'])
        if frame in inputs: assert pair == inputs[frame]
        inputs[frame] = pair
        assert pair[0]['shape'] == pair[1]['shape'] == row['left_received']['shape'] == row['right_received']['shape']
        raw_bytes = math.prod(pair[0]['shape']) * 2
        assert row['raw_RGB8_bits'] == 8 * raw_bytes and row['actual_raw_to_serialized_ratio'] == raw_bytes / row['source_bytes']
        totals[q][0] += raw_bytes; totals[q][1] += row['source_bytes']; rows += 1
    assert rows == info['actual_records'] == audit['rows'] == 6080
    ratios = {str(q): dict(raw_RGB8_bytes=r, full_wire_bytes=w, ratio=r / w) for q, (r, w) in totals.items()}
    assert ratios == info['pooled_ratios']
    choices = {}
    for target in (10, 30, 50):
        ranked = sorted((abs(math.log(r / w / target)), q) for q, (r, w) in totals.items())
        error, q = ranked[0]
        choices[str(target)] = dict(quality=q, measured_ratio=totals[q][0] / totals[q][1], absolute_log_residual=error)
    assert choices == audit['selected_qualities'] == info['selected_qualities']
    result = dict(state='passed_all6080_transferred_byte_records_and_rate_choices', checked_unix=time.time(),
        manifest_sha256=sha(manifest_path), audit_sha256=sha(audit_path), records_sha256=sha(raw),
        verifier_sha256=sha(Path(__file__)), rows=6080, selected_qualities=choices,
        source_PNGs_replayed_locally=False, candidate_wires_reencoded=False, KITTI_AP_measured=False,
        scope='Full transferred records and independent pooled ratio/selection; native PNG/source checks executed on sheng')
    with output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(state=result['state'], selected=choices)), flush=True)


if __name__ == '__main__': main()
