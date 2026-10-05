"""Every transferred wire/cache/control artifact and unclipped float32 pair."""
import argparse
import io
import json
import math
from pathlib import Path
import struct
import sys
import time
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/original-paper-scenarios/srcnn-native-cache-code'))
import common as c

NATIVE_ROOT = Path('/home/sheng/paper6')
PLAN = 'experiments/original-paper-scenarios/srcnn-native-cache-terminal-orchestration-001.md'


def literal_header(blob):
    assert len(blob) >= 32
    magic, version, algorithm, rate, reserved, h, w, lh, lw, nl, nr, crc = struct.unpack('>4sBBBBIIHHIII', blob[:32])
    assert (magic, version, algorithm, reserved) == (b'P6SR', 1, 1, 0) and rate in (10, 30, 50)
    assert h > 0 and w > 0 and (lh, lw) == (math.floor(h / math.sqrt(rate)), math.floor(w / math.sqrt(rate)))
    assert nl == nr == lh * lw * 3 > 0 and len(blob) == 32 + nl + nr
    assert zlib.crc32(blob[32:]) == crc
    return rate, h, w, lh, lw, nl, nr


def validate_float_pair(blob, row):
    assert c.r.sha_bytes(blob) == row['cache_sha256']
    tensors = c.r.tensor_pair(blob, row); count = 0
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        assert set(archive.files) == {'left', 'right'}
        for index, name in enumerate(('left', 'right')):
            value = archive[name]
            assert value.dtype == np.dtype('<f4') and list(value.shape) == row['native_hw'] + [3]
            assert np.isfinite(value).all() and c.r.describe(value) == row['arrays'][name]
            assert np.array_equal(tensors[index][0].numpy().transpose(1, 2, 0), value)
            assert row['ranges'][index] == dict(minimum=float(value.min()), maximum=float(value.max()),
                out_of_range_fraction=float(np.mean((value < 0) | (value > 1))))
            count += value.size
    return count


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--scope', choices=('engineering', 'main'), required=True)
    args = parser.parse_args(); prefix = c.prefix(args.scope)
    path = ROOT / 'data/provenance' / (prefix + '-closure.json')
    output = ROOT / 'data/provenance' / (prefix + '-local-verification.json'); assert not output.exists()
    closure = c.read(path)
    assert closure['state'] == 'closed_all_native_SRCNN_received_pairs_actual_terminal'
    assert closure['actual_terminal'] and closure['scope'] == args.scope
    source = c.CPU_gate(); assert closure['sources'] == source
    identity = {name: c.sha(ROOT / name) for name in (PLAN,
        'data/provenance/srcnn-native-cache-cycle-001.py', 'data/provenance/verify-srcnn-native-cache-local-001.py')}
    assert closure['orchestration_sources'] == identity
    expected_n = 2 if args.scope == 'engineering' else 3769
    expected_pairs = 3 * expected_n; expected_artifacts = 16 + 6 * expected_n
    assert closure['pairs'] == expected_pairs and len(closure['artifacts_sha256']) == expected_artifacts
    mapping = closure['native_to_local_artifacts']; assert set(mapping) == set(closure['artifacts_sha256'])
    assert len(set(mapping.values())) == expected_artifacts
    def local(native):
        p = Path(native)
        if p.is_relative_to(NATIVE_ROOT):
            relative = str(p.relative_to(NATIVE_ROOT))
        else:
            assert p.is_relative_to(c.NATIVE / prefix)
            relative = 'data/engineering/' + prefix + '-native-transfer/' + str(p).lstrip('/')
        assert mapping[relative] == native
        target = ROOT / relative; assert target.resolve().is_relative_to(ROOT)
        return target
    # Verify every actual file before interpreting reports or accepting any row.
    for relative, digest in closure['artifacts_sha256'].items():
        target = local(mapping[relative]); assert c.sha(target) == digest
    parent_path = NATIVE_ROOT / 'data/runs' / (prefix + '-encode.json')
    launch_path = NATIVE_ROOT / 'data/runs' / (prefix + '-launch.json')
    cycle_path = NATIVE_ROOT / 'data/runs' / (prefix + '-cycle.json')
    controller_log = NATIVE_ROOT / 'logs' / (prefix + '-controller.log')
    required = {str(p) for p in (parent_path, launch_path, cycle_path, controller_log)}
    parent, launch, cycle = [c.read(local(str(p))) for p in (parent_path, launch_path, cycle_path)]
    ids = parent['frame_ids']; assert len(ids) == len(set(ids)) == expected_n
    if args.scope == 'engineering':
        assert ids == ['000000', '000003']
    else:
        assert c.r.sha_bytes('\n'.join(ids).encode()) == c.VAL_SHA
    assert parent['state'] == 'finished_all_three_source_wire_conditions'
    assert parent['scope'] == args.scope and parent['pairs'] == expected_pairs and not parent['GPU_used']
    assert launch['state'] == 'launched_once' and cycle['state'] == 'finished_six_commands_pending_actual_terminal_closure'
    assert launch['pid'] == cycle['pid'] == closure['pid']
    assert launch['scope'] == cycle['scope'] == args.scope
    assert launch['orchestration_sources'] == cycle['orchestration_sources'] == identity
    assert parent['sources'] == launch['sources'] == cycle['sources'] == source
    assert parent['dependencies'] == launch['dependencies'] == cycle['dependencies'] == closure['dependencies']
    parent_digest = c.sha(local(str(parent_path)))
    assert launch['encoding_manifest_sha256'] == cycle['encoding_manifest_sha256'] == parent_digest
    train_scope = 'engineering' if args.scope == 'engineering' else 'formal'
    train_prefix = 'srcnn-kitti-' + train_scope + '-seed17-002'
    train_closure = ROOT / 'data/provenance' / (train_prefix + '-closure.json')
    train_proof = ROOT / 'data/provenance' / (train_prefix + '-local-verification.json')
    assert c.sha(train_closure) == closure['dependencies']['training_closure_sha256']
    assert c.sha(train_proof) == closure['dependencies']['training_local_proof_sha256']
    assert c.sha(ROOT / 'data/provenance/srcnn-float-reception-transferred-verification-001.json') == closure['dependencies']['float_CPU_local_proof_sha256']
    assert c.read(train_closure)['actual_terminal'] and c.read(train_proof)['sources'] == source['training']
    if args.scope == 'main':
        proof = c.read(train_proof)
        assert proof['state'] == 'passed_all111360_transferred_SRCNN_training_records_and_terminal_artifacts'
        assert proof['updates'] == 111360 and proof['independently_regenerated_view_and_patch_streams']
        assert proof['verifier_sha256'] == c.sha(ROOT / 'data/provenance/verify-srcnn-kitti-complete-local-003.py')
        engineer = c.read(ROOT / 'data/provenance/srcnn-source-cache-engineering-001-local-verification.json')
        assert engineer['state'] == 'passed_all6_transferred_native_SRCNN_received_pairs_and28_artifacts'
        assert engineer['sources'] == source
        assert engineer['closure_sha256'] == c.sha(ROOT / 'data/provenance/srcnn-source-cache-engineering-001-closure.json')
    assert len(cycle['completed_commands']) == 6 and set(cycle['rates']) == {'10', '30', '50'}
    for index, command in enumerate(cycle['completed_commands']):
        mode = 'receive' if index % 2 == 0 else 'audit'; rate = (10, 30, 50)[index // 2]
        expected_tail = [str(NATIVE_ROOT / 'experiments/original-paper-scenarios/srcnn-native-cache-code' / (mode + '.py')),
                         '--scope', args.scope, '--rate', str(rate)]
        assert command['command'][1:] == expected_tail and command['returncode'] == 0
        expected_log = str(NATIVE_ROOT / 'logs' / (prefix + '-command-' + str(index) + '.log'))
        assert command['log'] == expected_log and c.sha(local(expected_log)) == command['log_sha256']
        required.add(expected_log)
    summaries = {}; pixels = 0; maximum_error = 0.
    for rate in (10, 30, 50):
        receipt_path = str(NATIVE_ROOT / 'data/runs' / (prefix + '-cr' + str(rate) + '-receive.json'))
        audit_path = str(NATIVE_ROOT / 'data/provenance' / (prefix + '-cr' + str(rate) + '-audit.json'))
        required.update((receipt_path, audit_path)); receipt, audit = c.read(local(receipt_path)), c.read(local(audit_path))
        item = cycle['rates'][str(rate)]
        assert item['manifest'] == receipt_path and item['audit'] == audit_path
        assert c.sha(local(receipt_path)) == item['manifest_sha256'] == audit['manifest_sha256']
        assert c.sha(local(audit_path)) == item['audit_sha256']
        assert receipt['state'] == 'finished_all_native_received_pairs'
        assert audit['state'] == 'passed_every_native_SRCNN_source_wire_float_pixel_and_readonly_state'
        assert receipt['sources'] == audit['sources'] == source and receipt['dependencies'] == audit['dependencies'] == closure['dependencies']
        assert receipt['scope'] == audit['scope'] == args.scope and receipt['rate'] == audit['rate'] == rate
        assert receipt['encoding_manifest_sha256'] == audit['encoding_manifest_sha256'] == parent_digest
        assert receipt['ordered_ids'] == ids and len(receipt['frames']) == len(audit['rows']) == expected_n
        assert audit['pairs'] == expected_n and receipt['data_barrier'] and receipt['received_header_only_geometry']
        assert receipt['no_clean_GT_calibration_or_detector'] and not receipt['TF32'] and not receipt['mixed_precision']
        assert receipt['CPU_threads'] == 2 and receipt['cudnn_deterministic']
        assert receipt['source_only_PHY_uses'] is None and receipt['source_only_PHY_energy'] is None
        trained = c.read(ROOT / 'data/runs' / (train_prefix + '-cr' + str(rate) + '.json'))
        assert receipt['checkpoint_sha256'] == trained['checkpoint_sha256'] and trained['sources'] == source['training']
        assert receipt['model_initial_state'] == receipt['model_final_state'] == audit['readonly_states'] == trained['final_state']
        assert receipt['peak_reserved_bytes'] + 2 * 2**30 <= receipt['physical_free_before_bytes']
        assert audit['peak_reserved_bytes'] + 2 * 2**30 <= audit['physical_free_before_bytes']
        raw_total = wire_total = count = 0; rate_error = 0.
        condition = parent['conditions'][str(rate)]
        assert [row['frame_id'] for row in condition['rows']] == ids
        for frame, encoded, received, checked in zip(ids, condition['rows'], receipt['frames'], audit['rows']):
            assert frame == received['frame_id'] == checked['frame_id']
            wire_path = str(c.NATIVE / prefix / ('cr' + str(rate)) / 'wire' / (frame + '.p6sr'))
            cache_path = str(c.NATIVE / prefix / ('cr' + str(rate)) / 'received' / (frame + '.npz'))
            assert encoded['wire_path'] == received['wire_path'] == wire_path and received['cache_path'] == cache_path
            required.update((wire_path, cache_path))
            wire = local(wire_path).read_bytes(); blob = local(cache_path).read_bytes()
            assert c.r.sha_bytes(wire) == encoded['wire_sha256'] == received['wire_sha256'] == checked['wire_sha256']
            assert c.r.sha_bytes(blob) == received['cache_sha256'] == checked['cache_sha256']
            r, h, w, lh, lw, nl, nr = literal_header(wire)
            assert r == rate and received['native_hw'] == [h, w] and received['low_hw'] == [lh, lw]
            meta = encoded['source_metadata']
            for record in (meta, received):
                assert record['nominal_compression'] == rate and record['source_bytes'] == len(wire)
                assert record['framing_bytes'] == 32 and record['payload_bytes'] == [nl, nr]
                assert record['raw_RGB8_bits'] == h * w * 6 * 8
                assert record['actual_raw_to_wire_ratio'] == h * w * 6 / len(wire)
            assert received['received_header_only_geometry'] and received['output_unclipped']
            assert received['source_only_PHY_uses'] is None and received['source_only_PHY_energy'] is None
            assert received['readonly_states'] == trained['final_state'] and received['arrays'] == checked['arrays']
            for camera in ('source_left', 'source_right'):
                assert encoded[camera]['RGB8']['shape'] == [h, w, 3] and encoded[camera]['RGB8']['dtype'] == '|u1'
            assert checked['independent_max_abs_error'] <= 3e-5
            rate_error = max(rate_error, checked['independent_max_abs_error'])
            count += validate_float_pair(blob, received); raw_total += h * w * 6; wire_total += len(wire)
        assert audit['RGB_values'] == count and audit['maximum_RGB_error'] == rate_error
        assert condition['raw_RGB8_bytes'] == raw_total and condition['wire_bytes'] == audit['wire_bytes'] == wire_total
        assert condition['actual_pooled_raw_to_wire_ratio'] == audit['actual_pooled_raw_to_wire_ratio'] == raw_total / wire_total
        summaries[str(rate)] = dict(raw_RGB8_bytes=raw_total, wire_bytes=wire_total,
                                   actual_pooled_raw_to_wire_ratio=raw_total / wire_total, RGB_values=count)
        pixels += count; maximum_error = max(maximum_error, rate_error)
    assert required == set(mapping.values()) and pixels == closure['RGB_values']
    result = dict(state='passed_all' + str(expected_pairs) + '_transferred_native_SRCNN_received_pairs_and' + str(expected_artifacts) + '_artifacts',
        checked_unix=time.time(), scope=args.scope, pairs=expected_pairs, RGB_values=pixels,
        artifacts_verified=expected_artifacts, closure_sha256=c.sha(path), sources=source,
        dependencies=closure['dependencies'], orchestration_sources=identity, verifier_sha256=c.sha(__file__),
        summaries=summaries, maximum_native_independent_RGB_error=maximum_error,
        limitation='Every transferred native wire/cache/control artifact and float32 pixel identity checked; source PNG and model pixel replay native only, no quality, radio or AP')
    with output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(state=result['state'], pairs=expected_pairs, artifacts=expected_artifacts)))


if __name__ == '__main__': main()
