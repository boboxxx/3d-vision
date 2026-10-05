"""Complete cross-host CPU cache execution-contract evidence; no pixel replay."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def main():
    output = ROOT / 'data/provenance/srcnn-native-cache-CPU-transferred-verification-001.json'; assert not output.exists()
    paths = [ROOT / 'data/engineering' / ('srcnn-native-cache-' + host + '-CPU-002.json') for host in ('local', 'sheng')]
    reports = [read(path) for path in paths]; source = reports[0]['sources']
    for group in source.values():
        for path,digest in group.items(): assert sha(ROOT / path) == digest
    assert len(source['cache']) == 9 and len(source['training']) == 20
    initial_closure = read(ROOT / 'data/provenance/srcnn-kitti-engineering-seed17-002-closure.json')
    assert initial_closure['actual_terminal'] and initial_closure['sources'] == source['training']
    different = 0
    for report in reports:
        assert report['state'] == 'passed_five_native_SRCNN_cache_CPU_families'
        assert report['families'] == 5 and report['views'] == 6 and report['RGB_values'] == 29106
        assert report['source_geometry'] == [33,49] and report['sources'] == source
        assert report['maximum_RGB_error'] <= 3e-5 and report['malformed_header_CRC_checks'] == 24
        assert report['actual_file_open_barriers_checked'] and report['guaranteed_overshoot_cache_views'] == 2
        assert report['guaranteed_overshoot_RGB_values'] == 12
        assert len(report['records']) == 3 and [row['rate'] for row in report['records']] == [10,30,50]
        for row in report['records']:
            run = read(ROOT / 'data/runs' / ('srcnn-kitti-engineering-seed17-002-cr' + str(row['rate']) + '.json'))
            assert row['checkpoint_sha256'] == run['checkpoint_sha256'] and row['readonly_states'] == run['final_state']
            assert len(row['maximum_errors']) == len(row['ranges']) == 2 and max(row['maximum_errors']) <= 3e-5
            assert set(row['arrays']) == {'left','right'}
            for value in row['arrays'].values():
                assert value['dtype'] == '<f4' and value['shape'] == [33,49,3] and len(value['sha256']) == 64
    for local,native in zip(reports[0]['records'],reports[1]['records']):
        assert local['checkpoint_sha256'] == native['checkpoint_sha256'] and local['readonly_states'] == native['readonly_states']
        assert local['wire_sha256'] == native['wire_sha256'] and local['source_metadata'] == native['source_metadata']
        different += sum(local['arrays'][key] != native['arrays'][key] for key in ('left','right'))
    result = dict(state='passed_all12_transferred_native_SRCNN_cache_CPU_view_records', checked_unix=time.time(),
                  sources=source, reports={str(p.relative_to(ROOT)):sha(p) for p in paths}, families_each=5,
                  synthetic_views_each=6, synthetic_RGB_values_each=29106, explicit_overshoot_cache_views_each=2,
                  maximum_errors_each=[v['maximum_RGB_error'] for v in reports],
                  cross_host_output_hashes_different_views=different, source_wire_and_weights_match=True,
                  verifier_sha256=sha(__file__), limitation='All complete transferred CPU records and current source/weight lineage; CPU computations ran on each host, no native KITTI/GPU cache, quality or AP')
    with output.open('x') as stream: json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(state=result['state'], different_output_hashes=different)))


if __name__ == '__main__': main()
