"""Independent complete main cache metadata verification; raw pixels stay native."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'original-source-cache-main-001'
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
    closure_path = ROOT / 'data/provenance' / (PREFIX + '-closure.json'); closure = read(closure_path)
    assert closure['state'] == 'closed_actual_terminal_all_native_artifacts' and closure['actual_terminal']
    assert closure['scope'] == 'main' and closure['pairs'] == 22614 and closure['views'] == 45228
    assert len(closure['native_files']) == 45235 and len(closure['transferred_files']) == 19
    for p,h in closure['transferred_files'].items(): assert sha(native(p)) == h, p
    for group in closure['sources'].values():
        for p,h in group.items(): assert sha(ROOT / p) == h, p
    cycle = read(native('/home/sheng/paper6/data/runs/' + PREFIX + '-cycle.json'))
    launch = read(native('/home/sheng/paper6/data/runs/' + PREFIX + '-launch.json'))
    audit_path = '/home/sheng/paper6/data/provenance/' + PREFIX + '-audit.json'; audit = read(native(audit_path))
    assert cycle['state'] == 'finished_eight_commands_pending_terminal_closure'
    assert cycle['pid'] == launch['pid'] == closure['pid']
    assert len(cycle['completed_commands']) == 8 and all(x['returncode'] == 0 for x in cycle['completed_commands'])
    assert sha(ROOT / 'data/provenance/source-cache-cycle-001.py') == cycle['controller_sha256'] == launch['controller_sha256']
    assert cycle['audit_sha256'] == sha(native(audit_path))
    assert cycle['sources'] == launch['sources'] == audit['sources'] == closure['sources']
    assert audit['native_files_sha256'] == closure['native_files'] and audit['pairs'] == 22614 and audit['views'] == 45228
    assert audit['state'] == 'passed_all_six_complete_conditions_full_native_pixels_and_sources'
    encoding = read(native(NATIVE + '/encode.json')); ids = encoding['frame_ids']
    assert len(ids) == len(set(ids)) == 3769 and all(len(i) == 6 and i.isdecimal() for i in ids)
    # Reconstruct usual public text encodings; no frame/order changes permitted.
    split_blobs = [(separator.join(ids) + suffix).encode() for separator in ('\n', '\r\n') for suffix in ('', separator)]
    assert any(hashlib.sha256(blob).hexdigest() == '657ac4bcc1e156e5b106a4ca18e1f88e012787ea1d2b5d0adeea97fee903fa86' for blob in split_blobs)
    assert encoding['actual_encodes'] == 22614 and encoding['state'] == 'finished_all_six_complete_source_wires'
    assert encoding['no_GT_model_AP_or_radio'] and encoding['input_barrier']
    assert encoding['source_only_PHY_uses'] is None and encoding['source_only_PHY_energy'] is None
    assert encoding['sources'] == closure['sources'] and sha(native(NATIVE + '/encode.json')) == audit['encoding_manifest_sha256']
    assert sha(ROOT / 'experiments/original-paper-scenarios/source-cache-protocol-001.md') == encoding['protocol_sha256'] == audit['protocol_sha256']
    for p,h in encoding['predecessor'].items(): assert sha(ROOT / p) == h
    expected_env = dict(Python='3.10.15',NumPy='1.26.3',Pillow='10.2.0',libjpeg='6.2',libjpeg_turbo='3.0.1',OpenJPEG='2.5.0')
    assert encoding['environment'] == expected_env
    pids = set(); pairs = pixel_total = 0; source_pngs = {}; expected_native_files = {NATIVE+'/encode.json'}
    for key, codec, rate, parameter in CONDITIONS:
        receiver_path = NATIVE + '/received/' + key + '/receiver.json'; receiver = read(native(receiver_path))
        expected_native_files.add(receiver_path)
        source = encoding['conditions'][key]
        assert source['codec'] == codec and source['nominal_compression'] == rate and source['parameter'] == parameter
        assert receiver['state'] == 'finished_all_received_pairs' and receiver['condition'] == key and receiver['scope'] == 'main'
        assert receiver['frame_ids'] == ids and receiver['environment'] == expected_env and receiver['sources'] == closure['sources']
        assert receiver['input_barrier'] and not receiver['PNG_or_source_manifest_inputs'] and receiver['dimensions_from_received_stream_headers']
        assert receiver['HWC_uint8_no_resize'] and receiver['no_model_AP_or_radio']
        assert receiver['pid'] != encoding['pid'] and receiver['pid'] not in pids; pids.add(receiver['pid'])
        assert [x['frame_id'] for x in source['records']] == [x['frame_id'] for x in receiver['records']] == ids
        raw_sum = wire_sum = pixels = 0
        for s,r in zip(source['records'],receiver['records']):
            frame=s['frame_id']; e=s['evidence']; path=NATIVE+'/source/'+key+'/'+frame+'.p6sb'
            assert s['condition'] == key and s['codec'] == codec and s['parameter'] == parameter and s['nominal_compression'] == rate
            assert s['wire_path'] == r['wire_path'] == path and e['wire_sha256'] == r['wire_sha256'] == closure['native_files'][path]
            assert e['codec'] == codec and e['parameter'] == parameter and e['framing_bytes'] == 20
            nl,nr=e['codestream_bytes']; nbytes=e['source_bytes']; assert nl>0 and nr>0 and nbytes==20+nl+nr and e['source_bits']==8*nbytes
            options=(dict(quality=parameter,subsampling=2,optimize=False,progressive=False) if codec=='jpeg' else
                     dict(quality_mode='rates',quality_layers=[parameter],irreversible=True,mct=1,no_jp2=False))
            assert e['options'] == options
            h,w=r['public_hw']; count=h*w*3*2; assert h>0 and w>0
            for name in ('left','right'):
                assert r['arrays'][name]['dtype']=='|u1' and r['arrays'][name]['shape']==[h,w,3]
            assert e['left_received']==r['arrays']['left'] and e['right_received']==r['arrays']['right']
            assert e['left_input']['shape']==e['right_input']['shape']==[h,w,3]
            assert e['raw_RGB8_bits']==count*8 and e['actual_raw_to_serialized_ratio']==count/nbytes
            assert set(s['source_images_sha256']) == {f'/mnt/d/paper6/data/kitti/training/{v}/{frame}.png' for v in ('image_2','image_3')}
            for p,digest in s['source_images_sha256'].items():
                assert p not in source_pngs or source_pngs[p]==digest; source_pngs[p]=digest
            cache=NATIVE+'/received/'+key+'/'+frame+'.npz'; assert r['cache_path']==cache and closure['native_files'][cache]==r['cache_sha256']
            expected_native_files.update((path,cache)); raw_sum+=count; wire_sum+=nbytes; pixels+=count; pairs+=1
        summary=audit['conditions'][key]
        assert summary['pairs']==3769 and summary['raw_RGB8_bytes']==raw_sum and summary['full_wire_bytes']==wire_sum
        assert summary['pooled_raw_to_wire_ratio']==raw_sum/wire_sum and summary['full_pixel_values_exact']==pixels
        pixel_total+=pixels
    assert pairs==22614 and len(pids)==6 and len(source_pngs)==7538
    assert set(closure['native_files'])==expected_native_files and audit['full_pixel_values_exact']==pixel_total
    result=dict(state='passed_all22614_transferred_pair_records_and_native_audit_metadata',checked_unix=time.time(),
                pairs=pairs,views=45228,sources=closure['sources'],closure_sha256=sha(closure_path),
                native_files=45235,transferred_files=19,conditions=audit['conditions'],
                limitation='Full wire CRC and cache pixels replayed on sheng, not locally; no detector AP or radio performance')
    with output.open('x') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(dict(state=result['state'],pairs=pairs)))


if __name__ == '__main__': main()
