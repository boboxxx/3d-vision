"""Independent transferred metadata/received-byte verification; no raw-array claim."""
import hashlib
import json
from pathlib import Path
import struct
import time
import zlib

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'ecsic-neural-reception-native-001'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def main():
    closure_path = ROOT / 'data/provenance' / (PREFIX + '-closure.json')
    closure = read(closure_path)
    assert closure['state'] == 'closed_actual_terminal_all20_outcomes11_neural_crop_pairs'
    assert closure['actual_terminal'] and closure['ps_returncode'] == 1 and len(closure['actual_ps'].splitlines()) <= 1
    assert not closure['local_raw_arrays_transferred'] and not closure['KITTI_AP_measured']
    for p, digest in closure['artifacts_sha256'].items(): assert sha(ROOT / p) == digest
    for group in closure['sources'].values():
        for p, digest in group.items(): assert sha(ROOT / p) == digest
    previous = closure['predecessor']
    for p, digest in {**previous['locked_evidence_sha256'], **previous['transfer_proofs_sha256']}.items():
        assert sha(ROOT / p if '/' in p else ROOT / 'data/provenance' / p) == digest
    for p, digest in previous['received_files_sha256'].items(): assert sha(ROOT / p) == digest
    assert len(previous['received_files_sha256']) == 22
    cycle = read(ROOT / 'data/runs' / (PREFIX + '-cycle.json'))
    audit = read(ROOT / 'data/provenance' / (PREFIX + '-audit.json'))
    snapshot = ROOT / 'data/engineering' / PREFIX
    manifest = read(snapshot / 'manifest.json')
    assert sha(snapshot / 'manifest.json') == closure['manifest_sha256'] == cycle['manifest_sha256'] == audit['producer_manifest_sha256']
    assert closure['pid'] == cycle['pid'] and closure['sources'] == manifest['sources'] == cycle['sources']
    assert previous == manifest['predecessor'] == cycle['predecessor']
    assert (audit['packets'], audit['received'], audit['erased'], audit['full_arrays_exact'], audit['cropped_views_exact'], audit['distinct_successful_child_processes']) == (20, 11, 9, 198, 22, 22)
    original = read(ROOT / 'data/engineering/artemis-ecsic-digital-CPU-001.json')
    assert len(original['packets']) == len(manifest['packets']) == 20
    assert len(closure['snapshot_map']) == 45 and len(closure['native_artifacts_sha256']) == 67
    for p, item in closure['snapshot_map'].items():
        assert sha(ROOT / p) == item['sha256'] == audit['metadata_artifacts_sha256'][item['native_path']]
        assert item['sha256'] == closure['native_artifacts_sha256'][item['native_path']]
    pids = set(); end = manifest['started_unix']; received = erased = 0; source_records = {}
    for source, packet in zip(original['packets'], manifest['packets']):
        for key in ('index', 'name', 'source', 'configuration', 'channel', 'snr_db', 'attempted_uses', 'attempted_energy', 'received_paths'):
            assert packet[key] == source[key]
        assert packet['original_reception'] == source['reception']
        target = snapshot / source['name']
        if source['reception']['state'] == 'erased':
            assert packet['state'] == 'erased_preserved_no_neural_or_crop' and packet['neural_calls'] == packet['crop_calls'] == 0
            assert packet['output_dir'] is None and not target.exists(); erased += 1; continue
        received += 1
        for execution in (packet['receiver_execution'], packet['crop_execution']):
            assert execution['pid'] not in pids and execution['returncode'] == 0
            assert end <= execution['started_unix'] <= execution['ended_unix']
            pids.add(execution['pid']); end = execution['ended_unix']
        payload = (ROOT / source['received_paths']['payload']).read_bytes()
        wire = (ROOT / source['received_paths']['wire']).read_bytes()
        magic, version, codec, reserved, length, second, crc = struct.unpack_from('>4sBBHIII', wire)
        assert (magic, version, codec, reserved, length, second, crc) == (b'P6SB', 2, 3, 0, len(payload), 0, zlib.crc32(payload))
        assert wire[20:] == payload and zlib.crc32(payload[:-4]) == struct.unpack('>I', payload[-4:])[0]
        magic, version, count, oh, ow, ph, pw, weight, config, cdf = struct.unpack_from('>4sBB4H32s32s32s', payload)
        assert (magic, version, count) == (b'P6EC', 1, 4)
        receiver, crop = read(target / 'receiver.json'), read(target / 'crop.json')
        assert receiver['state'] == crop['state'] == 'passed'
        assert receiver['container_sha256'] == crop['container_sha256'] == sha(ROOT / source['received_paths']['payload'])
        assert receiver['model_sha256'] == weight.hex() and receiver['config_sha256'] == config.hex() and receiver['CDF_sha256'] == cdf.hex()
        assert receiver['public_dimensions'] == dict(original_hw=[oh, ow], padded_hw=[ph, pw])
        assert receiver['state_count'] == len(receiver['full_states_before']) == 225 and receiver['full_states_before'] == receiver['full_states_after']
        assert receiver['calls'] == dict(E=0, HE=0, HD=1, D=1) and receiver['no_parameter_gradients'] and receiver['source_NPZ_read_barrier']
        assert len(receiver['arrays']) == 18 and receiver['output_sha256'] == packet['files_sha256']['reconstructed.npz'] == crop['reconstructed_sha256']
        assert crop['receiver_report_sha256'] == sha(target / 'receiver.json') and crop['output_sha256'] == packet['files_sha256']['received-RGB-FP32.npz']
        assert crop['original_hw'] == [oh, ow] and crop['padded_hw'] == [ph, pw]
        assert crop['unclipped_FP32'] and crop['no_resize_or_RGB8_conversion'] and crop['source_NPZ_read_barrier']
        assert set(crop['arrays']) == {'left', 'right'}
        for value in crop['arrays'].values(): assert value['shape'] == [1, 3, oh, ow] and value['dtype'] == '<f4' and value['finite']
        signature = (receiver['arrays'], receiver['full_states_before'], crop['arrays'])
        if source['source'] in source_records: assert signature == source_records[source['source']]
        source_records[source['source']] = signature
        for name in ('receiver.json', 'crop.json', 'neural.log', 'crop.log'): assert sha(target / name) == packet['files_sha256'][name]
    assert (received, erased, len(pids)) == (11, 9, 22) and end <= manifest['ended_unix']
    output = ROOT / 'data/provenance' / (PREFIX + '-local-verification-001.json')
    result = dict(state='passed_all_transferred_metadata_and22_actual_received_files', checked_unix=time.time(),
        closure_sha256=sha(closure_path), verifier_sha256=sha(Path(__file__)),
        artifacts_verified=len(closure['artifacts_sha256']), metadata_snapshots=45, received_files=22,
        packets=20, received=11, erased=9, distinct_child_processes=22,
        raw_arrays_replayed_locally=False, KITTI_AP_measured=False)
    with output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
