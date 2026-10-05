"""Independent full20 outcomes/225states/18arrays/crop audit; no producer imports."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
import zlib

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE_B = Path('/mnt/d/paper6/runs/ecsic-entropy-stageB-002')
WEIGHT = Path('/mnt/d/paper6/assets/ecsic-cs001-001/model.pt')
WEIGHT_SHA = 'e66b55e2f888f5d113ea6c6414640c7486280d6e18838ba77c650720a5636052'
CONFIG_SHA = 'ba02cc1239b755a26ca0eb920c3014908c97c45fa210c447c47e96f7527102f8'
CDF_SHA = '507d22e35fb0bb605f05a3312600a8aff3ff10f1ec1c4964358cb390e35d56e5'
MANIFEST_SHA = 'f265c56a7fdcc3e98fde0ff580758479aa50036d4b474fa949ec7cbc605227d3'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''): h.update(part)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def describe(a):
    a = np.ascontiguousarray(a)
    return dict(shape=list(a.shape), dtype=str(a.dtype), sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                finite=bool(np.isfinite(a).all()), min=float(a.min()), max=float(a.max()))


def inner_header(blob):
    assert len(blob) >= 182 and zlib.crc32(blob[:-4]) == struct.unpack('>I', blob[-4:])[0]
    h = struct.Struct('>4sBB4H32s32s32s'); values = h.unpack_from(blob)
    magic, version, count, oh, ow, ph, pw, weight, config, cdf = values
    assert (magic, version, count) == (b'P6EC', 1, 4)
    assert (weight.hex(), config.hex(), cdf.hex()) == (WEIGHT_SHA, CONFIG_SHA, CDF_SHA)
    assert 0 < oh <= ph <= 2048 and 0 < ow <= pw <= 4096 and ph % 32 == pw % 32 == 0 and ph-oh < 32 and pw-ow < 32
    offset = h.size; body = 0
    for index in range(4):
        i, symbols, rans, escape = struct.unpack_from('>BIII', blob, offset); offset += 13
        divisor = 32 if index < 2 else 8
        assert i == index and symbols == 48*(ph//divisor)*(pw//divisor) and rans >= 4 and escape <= symbols*4
        body += rans + escape
    assert offset + body + 4 == len(blob)
    return [oh, ow], [ph, pw]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    assert not args.output.exists() and not torch.cuda.is_available()
    torch.set_num_threads(2)
    producer = args.directory / 'manifest.json'; bridge = read(producer)
    assert bridge['state'] == 'finished20_outcomes11_neural_crop_pairs_independent_audit_pending'
    assert bridge['fresh_neural_processes'] == bridge['fresh_crop_processes'] == 11 and len(bridge['packets']) == 20
    original_path = ROOT / 'data/engineering/artemis-ecsic-digital-CPU-001.json'
    assert sha(original_path) == MANIFEST_SHA
    original = read(original_path)
    assert [x['name'] for x in bridge['packets']] == [x['name'] for x in original['packets']]
    for key, files in bridge['sources'].items(): assert all(sha(ROOT / p) == h for p, h in files.items())
    previous = bridge['predecessor']
    for p, h in {**previous['locked_evidence_sha256'], **previous['transfer_proofs_sha256']}.items():
        path = ROOT / p if '/' in p else ROOT / 'data/provenance' / p
        assert sha(path) == h
    assert len(previous['received_files_sha256']) == 22
    assert all(sha(ROOT / p) == h for p, h in previous['received_files_sha256'].items())
    phaseB = read(SOURCE_B / 'manifest.json')
    assert sha(SOURCE_B / 'manifest.json') == original['stageB_manifest_sha256']
    assert sha(WEIGHT) == WEIGHT_SHA
    weights = torch.load(WEIGHT, map_location='cpu', weights_only=True)
    assert len(weights) == 225
    expected_states = {k: describe(v.detach().cpu().numpy()) for k, v in weights.items()}
    del weights
    packet_audits = []; native_artifacts = {str(producer): sha(producer)}; metadata_artifacts = {str(producer): sha(producer)}
    accepted_names = set(); full_arrays = crop_arrays = 0
    child_PIDs = set(); previous_end = bridge['started_unix']
    expected_array_names = {n + '_' + s for n in ('z_left', 'z_right', 'y_left', 'y_right') for s in ('loc', 'scale', 'hat', 'symbols')} | {'pred_left', 'pred_right'}
    for source, item in zip(original['packets'], bridge['packets']):
        for key in ('index', 'name', 'source', 'configuration', 'channel', 'snr_db', 'attempted_uses', 'attempted_energy', 'received_paths'):
            assert item[key] == source[key]
        assert item['original_reception'] == source['reception']
        target = args.directory / source['name']
        if source['reception']['state'] == 'erased':
            assert item['state'] == 'erased_preserved_no_neural_or_crop' and item['neural_calls'] == item['crop_calls'] == 0
            assert item['output_dir'] is None and not target.exists()
            packet_audits.append(dict(name=source['name'], state='erased_preserved', reason=source['reception']['erasure_reason'])); continue
        accepted_names.add(source['name'])
        assert item['state'] == 'neural_and_crop_finished_independent_audit_pending' and item['neural_calls'] == item['crop_calls'] == 1
        for execution in (item['receiver_execution'], item['crop_execution']):
            assert type(execution['pid']) is int and execution['pid'] > 0 and execution['pid'] not in child_PIDs
            assert execution['returncode'] == 0 and previous_end <= execution['started_unix'] <= execution['ended_unix']
            child_PIDs.add(execution['pid']); previous_end = execution['ended_unix']
        assert Path(item['output_dir']) == target
        accepted = source['received_paths']; payload = (ROOT / accepted['payload']).read_bytes(); wire = (ROOT / accepted['wire']).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == accepted['payload_sha256'] and hashlib.sha256(wire).hexdigest() == accepted['wire_sha256']
        magic, version, codec, reserved, length, second, crc = struct.unpack_from('>4sBBHIII', wire)
        assert (magic, version, codec, reserved, second) == (b'P6SB', 2, 3, 0, 0) and length == len(payload)
        assert wire[20:] == payload and zlib.crc32(payload) == crc
        original_hw, padded_hw = inner_header(payload)
        receiver = read(target / 'receiver.json'); crop = read(target / 'crop.json')
        assert receiver['state'] == crop['state'] == 'passed' and receiver['container_sha256'] == crop['container_sha256'] == accepted['payload_sha256']
        assert receiver['container_bytes'] == len(payload) and receiver['public_dimensions'] == dict(original_hw=original_hw, padded_hw=padded_hw)
        assert receiver['state_count'] == 225 and receiver['full_states_before'] == receiver['full_states_after'] == expected_states
        assert receiver['no_parameter_gradients'] and receiver['source_NPZ_read_barrier'] and receiver['device'] == 'cpu'
        assert receiver['calls'] == dict(E=0, HE=0, HD=1, D=1) and receiver['torch_threads'] == 2
        assert receiver['model_sha256'] == WEIGHT_SHA and receiver['config_sha256'] == CONFIG_SHA and receiver['CDF_sha256'] == CDF_SHA
        base = SOURCE_B / source['source']; reference = read(base / 'receiver.json')
        assert sha(base / 'receiver.json') == phaseB['cases'][source['source']]['files']['receiver.json']
        assert sha(base / 'reconstructed.npz') == phaseB['cases'][source['source']]['files']['reconstructed.npz'] == reference['output_sha256']
        assert receiver['versions'] == reference['versions'] == bridge['reference_identities'][source['source']]['versions']
        assert receiver['official_sources'] == reference['official_sources']
        assert receiver['derived_scale_indices'] == reference['derived_scale_indices']
        assert receiver['arrays'] == reference['arrays']
        assert sha(target / 'reconstructed.npz') == receiver['output_sha256'] == crop['reconstructed_sha256']
        assert sha(target / 'receiver.json') == crop['receiver_report_sha256']
        assert sha(target / 'received-RGB-FP32.npz') == crop['output_sha256']
        assert crop['original_hw'] == original_hw and crop['padded_hw'] == padded_hw
        assert crop['unclipped_FP32'] and crop['no_resize_or_RGB8_conversion'] and crop['source_NPZ_read_barrier']
        assert crop['crop'] == 'top_left_public_header_NCHW[:, :, :h, :w]'
        assert crop['allowed_input_NPZ'] == str((target / 'reconstructed.npz').resolve())
        with np.load(target / 'reconstructed.npz', allow_pickle=False) as actual, np.load(base / 'reconstructed.npz', allow_pickle=False) as baseline, np.load(target / 'received-RGB-FP32.npz', allow_pickle=False) as pixels:
            assert set(actual.files) == set(baseline.files) == expected_array_names and set(pixels.files) == {'left', 'right'}
            for name in expected_array_names:
                assert np.array_equal(actual[name], baseline[name]) and actual[name].dtype == baseline[name].dtype
                assert describe(actual[name]) == receiver['arrays'][name]; full_arrays += 1
            h, w = original_hw
            for side in ('left', 'right'):
                expected = actual['pred_' + side][:, :, :h, :w]
                assert pixels[side].shape == (1, 3, h, w) and pixels[side].dtype == np.float32 and np.array_equal(pixels[side], expected)
                described = describe(pixels[side]); described['dtype'] = pixels[side].dtype.str
                assert described == crop['arrays'][side]; crop_arrays += 1
        assert set(item['files_sha256']) == {'neural.log', 'receiver.json', 'reconstructed.npz', 'crop.log', 'crop.json', 'received-RGB-FP32.npz'}
        for file, digest in item['files_sha256'].items():
            path = target / file; assert sha(path) == digest; native_artifacts[str(path)] = digest
            if path.suffix in ('.json', '.log'): metadata_artifacts[str(path)] = digest
        assert item['receiver_command'] == [item['receiver_command'][0], str(ROOT / 'experiments/original-paper-scenarios/ecsic-entropy-code/native.py'), '--receiver', '--container', str(ROOT / accepted['payload']), '--directory', str(target)]
        assert item['crop_command'] == [item['crop_command'][0], str(ROOT / 'experiments/original-paper-scenarios/ecsic-neural-reception-code/crop.py'), '--container', str(ROOT / accepted['payload']), '--directory', str(target)]
        packet_audits.append(dict(name=source['name'], state='passed225_states18_arrays_exact_public_crop', full_arrays=18, cropped_views=2))
    assert {p.name for p in args.directory.iterdir() if p.is_dir()} == accepted_names and len(accepted_names) == 11
    assert full_arrays == 198 and crop_arrays == 22
    assert len(child_PIDs) == 22 and previous_end <= bridge['ended_unix']
    result = dict(state='passed_all20_outcomes11_causal_neural_and_exact_crops', checked_unix=time.time(),
                  producer_manifest_sha256=sha(producer), predecessor_manifest_sha256=MANIFEST_SHA,
                  packets=20, received=11, erased=9, full_arrays_exact=198, cropped_views_exact=22,
                  all225_states_exact_and_readonly=True, no_source_encoder_or_erasures_fallback=True,
                  distinct_successful_child_processes=22,
                  packet_audits=packet_audits, native_artifacts_sha256=native_artifacts, metadata_artifacts_sha256=metadata_artifacts,
                  physical_attempts_resources_unchanged=True, KITTI_AP_measured=False,
                  scope='Independent raw18-array/225weight/publiccrop checks on sheng; no model forward in auditor; fixed two-source engineering, not BLER/AP')
    with args.output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(state=result['state'], received=11, erased=9, arrays=198, crops=22)), flush=True)


if __name__ == '__main__': main()
