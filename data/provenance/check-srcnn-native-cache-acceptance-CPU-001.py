"""Preparation-only actual six-wire parsing and float cache tamper checks."""
import importlib.util
import io
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('cache_acceptance', ROOT / 'data/provenance/verify-srcnn-native-cache-local-001.py')
v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)


def main():
    output = ROOT / 'data/engineering/srcnn-native-cache-acceptance-local-CPU-001.json'; assert not output.exists()
    prefix = 'srcnn-source-cache-engineering-001'
    encode = v.c.read(ROOT / 'data/runs' / (prefix + '-encode.json'))
    proof = v.c.read(ROOT / 'data/provenance' / (prefix + '-wire-local-verification.json'))
    assert proof['state'] == 'passed_all6_actual_transferred_SRCNN_engineering_wire_files_and8_artifacts'
    checks = 0; rejected = 0
    for rate in (10, 30, 50):
        for row in encode['conditions'][str(rate)]['rows']:
            path = ROOT / 'data/engineering' / (prefix + '-wire-transfer') / row['wire_path'].lstrip('/')
            blob = path.read_bytes(); assert v.c.sha(path) == row['wire_sha256']
            parsed = v.literal_header(blob); assert parsed[0] == rate
            assert list(parsed[1:3]) == row['source_metadata']['native_hw']; checks += 1
            for corrupt in (blob[:-1], blob[:-1] + bytes([blob[-1] ^ 1])):
                try: v.literal_header(corrupt)
                except AssertionError: rejected += 1
                else: raise AssertionError('malformed actual wire accepted')
    values = [np.array([[[-2., .5, 2.], [0., 1., -1.]]], dtype='<f4'),
              np.array([[[2., -2., 1.], [.25, -1., 0.]]], dtype='<f4')]
    blob = v.c.r.pack_pair(values)
    row = dict(cache_sha256=v.c.r.sha_bytes(blob), native_hw=[1, 2],
               arrays={name:v.c.r.describe(a) for name,a in zip(('left','right'),values)},
               ranges=[dict(minimum=float(a.min()), maximum=float(a.max()),
                            out_of_range_fraction=float(np.mean((a<0)|(a>1)))) for a in values])
    assert v.validate_float_pair(blob, row) == 12
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        assert all(np.array_equal(archive[name], a) for name,a in zip(('left','right'),values))
    for changed in (blob[:-1] + bytes([blob[-1] ^ 1]), v.c.r.pack_pair([a.clip(0,1) for a in values])):
        try: v.validate_float_pair(changed, row)
        except AssertionError: rejected += 1
        else: raise AssertionError('changed cache accepted under original identity')
    buffer = io.BytesIO(); np.savez(buffer, left=values[0].astype(np.float64), right=values[1].astype(np.float64))
    altered_row = dict(row, cache_sha256=v.c.r.sha_bytes(buffer.getvalue()))
    try: v.validate_float_pair(buffer.getvalue(), altered_row)
    except AssertionError: rejected += 1
    else: raise AssertionError('float64 cache accepted')
    altered_row = dict(row, ranges=[dict(row['ranges'][0], minimum=0.), row['ranges'][1]])
    try: v.validate_float_pair(blob, altered_row)
    except AssertionError: rejected += 1
    else: raise AssertionError('changed range metadata accepted')
    assert checks == 6 and rejected == 16
    files = ['data/provenance/srcnn-native-cache-cycle-001.py',
             'data/provenance/verify-srcnn-native-cache-local-001.py',
             'data/provenance/check-srcnn-native-cache-acceptance-CPU-001.py',
             'experiments/original-paper-scenarios/srcnn-native-cache-terminal-orchestration-001.md']
    result = dict(state='passed_preparation_actual6_wire_and_float_cache_acceptance_CPU_checks',
        checked_unix=time.time(), actual_transferred_wires=6, synthetic_float_values=12,
        rejected_corrupt_or_changed_cases=16, overshoot_values_preserved=True,
        sources=v.c.sources(), acceptance_sources={name:v.c.sha(ROOT/name) for name in files},
        limitation='CPU helper contracts only; no native GPU pair, controller dispatch, terminal closure, source quality or AP')
    with output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({key:result[key] for key in ('state','actual_transferred_wires','rejected_corrupt_or_changed_cases')}))


if __name__ == '__main__': main()
