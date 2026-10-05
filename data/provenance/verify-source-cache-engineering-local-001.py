"""Independent complete transferred manifests and real framed-wire verification."""
import hashlib
import json
from pathlib import Path
import struct
import time
import zlib

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'original-source-cache-engineering-001'
MIRROR = ROOT / 'data/engineering' / (PREFIX + '-transfer')
NATIVE = '/mnt/d/paper6/runs/' + PREFIX
CONDITIONS = [(f'{codec}-cr{rate}', codec, rate, {10:90,30:39,50:17}[rate] if codec == 'jpeg' else rate)
              for codec in ('jpeg', 'jpeg2000') for rate in (10,30,50)]


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def native(p): return MIRROR / p.lstrip('/')


def main():
    output = ROOT / 'data/provenance' / (PREFIX + '-local-verification.json')
    assert not output.exists()
    closure_path = ROOT / 'data/provenance' / (PREFIX + '-closure.json')
    closure = read(closure_path)
    assert closure['state'] == 'closed_actual_terminal_all_native_artifacts' and closure['actual_terminal']
    assert closure['scope'] == 'engineering' and closure['pairs'] == 12 and closure['views'] == 24
    assert len(closure['native_files']) == 31 and len(closure['transferred_files']) == 31
    for p, digest in closure['transferred_files'].items(): assert sha(native(p)) == digest, p
    for group in closure['sources'].values():
        for p, digest in group.items(): assert sha(ROOT / p) == digest, p
    cycle = read(native('/home/sheng/paper6/data/runs/' + PREFIX + '-cycle.json'))
    launch = read(native('/home/sheng/paper6/data/runs/' + PREFIX + '-launch.json'))
    audit = read(native('/home/sheng/paper6/data/provenance/' + PREFIX + '-audit.json'))
    assert cycle['state'] == 'finished_eight_commands_pending_terminal_closure'
    assert cycle['pid'] == launch['pid'] == closure['pid']
    assert len(cycle['completed_commands']) == 8 and all(x['returncode'] == 0 for x in cycle['completed_commands'])
    assert sha(ROOT / 'data/provenance/source-cache-cycle-001.py') == cycle['controller_sha256'] == launch['controller_sha256']
    assert cycle['sources'] == launch['sources'] == audit['sources'] == closure['sources']
    assert audit['native_files_sha256'] == closure['native_files'] and audit['pairs'] == 12 and audit['views'] == 24
    assert audit['state'] == 'passed_all_six_complete_conditions_full_native_pixels_and_sources'
    encoding = read(native(NATIVE + '/encode.json')); ids = ['000000','000003']
    assert encoding['frame_ids'] == ids and encoding['actual_encodes'] == 12
    assert encoding['state'] == 'finished_all_six_complete_source_wires' and encoding['no_GT_model_AP_or_radio']
    assert encoding['source_only_PHY_uses'] is None and encoding['source_only_PHY_energy'] is None
    assert encoding['sources'] == closure['sources'] and sha(native(NATIVE + '/encode.json')) == audit['encoding_manifest_sha256']
    assert sha(ROOT / 'experiments/original-paper-scenarios/source-cache-protocol-001.md') == encoding['protocol_sha256'] == audit['protocol_sha256']
    for p, digest in encoding['predecessor'].items(): assert sha(ROOT / p) == digest
    pids = set(); wire_total = pairs = 0
    for key, codec, rate, parameter in CONDITIONS:
        receiver = read(native(NATIVE + '/received/' + key + '/receiver.json'))
        source = encoding['conditions'][key]
        assert source['codec'] == codec and source['nominal_compression'] == rate and source['parameter'] == parameter
        assert receiver['state'] == 'finished_all_received_pairs' and receiver['condition'] == key
        assert receiver['frame_ids'] == ids and receiver['environment'] == encoding['environment'] and receiver['sources'] == closure['sources']
        assert receiver['input_barrier'] and not receiver['PNG_or_source_manifest_inputs'] and receiver['dimensions_from_received_stream_headers']
        assert receiver['HWC_uint8_no_resize'] and receiver['no_model_AP_or_radio']
        assert receiver['pid'] != encoding['pid'] and receiver['pid'] not in pids; pids.add(receiver['pid'])
        assert [x['frame_id'] for x in source['records']] == [x['frame_id'] for x in receiver['records']] == ids
        raw_sum = wire_sum = pixels = 0
        for s, r in zip(source['records'], receiver['records']):
            frame = s['frame_id']; e = s['evidence']; path = NATIVE + '/source/' + key + '/' + frame + '.p6sb'
            assert s['wire_path'] == r['wire_path'] == path
            blob = native(path).read_bytes(); magic, ver, code, reserved, nl, nr, crc = struct.unpack_from('>4sBBHIII', blob)
            assert (magic, ver, code, reserved) == (b'P6SB',1,1 if codec == 'jpeg' else 2,0)
            assert nl > 0 and nr > 0 and len(blob) == 20 + nl + nr and zlib.crc32(blob[20:]) == crc
            assert sha(native(path)) == e['wire_sha256'] == r['wire_sha256'] == closure['native_files'][path]
            assert e['source_bytes'] == len(blob) and e['source_bits'] == 8*len(blob) and e['codestream_bytes'] == [nl,nr]
            assert e['left_received'] == r['arrays']['left'] and e['right_received'] == r['arrays']['right']
            h,w = r['public_hw']; count = h*w*3*2
            for name in ('left','right'):
                assert r['arrays'][name]['dtype'] == '|u1' and r['arrays'][name]['shape'] == [h,w,3]
            assert e['raw_RGB8_bits'] == count*8 and e['actual_raw_to_serialized_ratio'] == count/len(blob)
            assert r['cache_path'] == NATIVE + '/received/' + key + '/' + frame + '.npz'
            assert closure['native_files'][r['cache_path']] == r['cache_sha256']
            raw_sum += count; wire_sum += len(blob); pixels += count; pairs += 1
        summary = audit['conditions'][key]
        assert summary['pairs'] == 2 and summary['raw_RGB8_bytes'] == raw_sum and summary['full_wire_bytes'] == wire_sum
        assert summary['pooled_raw_to_wire_ratio'] == raw_sum/wire_sum and summary['full_pixel_values_exact'] == pixels
        wire_total += wire_sum
    assert pairs == 12 and len(pids) == 6
    result = dict(state='passed_all_transferred_records_and_engineering_wires', checked_unix=time.time(), pairs=pairs,
                  wire_bytes=wire_total, sources=closure['sources'], closure_sha256=sha(closure_path),
                  transferred_files=31, native_pixel_audit_only=True,
                  limitation='Full native cache pixels verified on sheng; no local raw cache pixel replay or detector AP')
    with output.open('x') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(dict(state=result['state'],pairs=pairs)))


if __name__ == '__main__': main()
