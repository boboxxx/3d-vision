"""Complete transferred six-view CPU records, source and checkpoint lineage."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    output = ROOT / 'data/provenance/srcnn-float-reception-transferred-verification-001.json'
    assert not output.exists()
    paths = [ROOT / 'data/provenance' / ('srcnn-float-reception-cpu-' + host + '-002.json')
             for host in ('local', 'sheng')]
    reports = list(map(read, paths))
    closure_path = ROOT / 'data/provenance/srcnn-kitti-engineering-seed17-002-closure.json'
    closure = read(closure_path)
    assert closure['actual_terminal']
    selected = {}
    for report in reports:
        assert report['state'] == 'passed_six_trained_SRCNN_float_reception_CPU_families'
        assert report['families'] == report['views'] == 6 and report['pixels'] == 27648
        assert report['readonly_states_each'] == 6
        assert report['maximum_RGB_error'] < 3e-5 and report['maximum_model_error'] < 2e-5
        assert report['engineering_closure_sha256'] == sha(closure_path)
        assert report['training_sources'] == closure['sources']
        assert len(report['sources']) == 2
        for path, digest in dict(report['sources'], **report['training_sources']).items():
            assert sha(ROOT / path) == digest
        assert sha(ROOT / 'experiments/original-paper-scenarios/srcnn-float-reception-protocol-001.md') == report['protocol_sha256']
        assert [row['rate'] for row in report['records']] == [10, 30, 50]
        for row in report['records']:
            rate = row['rate']; run = read(ROOT / 'data/runs' / ('srcnn-kitti-engineering-seed17-002-cr' + str(rate) + '.json'))
            assert row['checkpoint_sha256'] == run['checkpoint_sha256']
            received = row['received']; sent = row['source_metadata']
            assert received['readonly_states'] == run['final_state'] and len(received['readonly_states']) == 6
            assert received['received_header_only_geometry'] and received['output_unclipped']
            assert received['native_hw'] == sent['native_hw'] == [32, 48]
            assert sent['nominal_compression'] == received['nominal_compression'] == rate
            assert row['source_bytes'] == sent['source_bytes'] == received['source_bytes']
            assert sent['framing_bytes'] == received['framing_bytes'] == 32
            assert sent['source_bits'] == 8 * sent['source_bytes']
            assert sent['raw_RGB8_bits'] == 8 * 32 * 48 * 3 * 2
            assert sent['actual_raw_to_wire_ratio'] == (32 * 48 * 3 * 2) / row['source_bytes']
            assert sent['source_only_PHY_uses'] is None and sent['source_only_PHY_energy'] is None
            assert received['source_only_PHY_uses'] is None and received['source_only_PHY_energy'] is None
            assert len(row['independent_model_max_abs_errors']) == len(row['independent_RGB_max_abs_errors']) == 2
            assert max(row['independent_model_max_abs_errors']) < 2e-5 and max(row['independent_RGB_max_abs_errors']) < 3e-5
            assert set(received['arrays']) == {'left', 'right'} and len(row['tensors']) == 2
            assert all(a['dtype'] == '<f4' and a['shape'] == [32, 48, 3] for a in received['arrays'].values())
            assert all(t['dtype'] == '<f4' and t['shape'] == [1, 3, 32, 48] for t in row['tensors'])
            selected[rate] = run['checkpoint_sha256']
    left, right = reports
    assert left['sources'] == right['sources'] and left['training_sources'] == right['training_sources']
    different = 0
    for local, native in zip(left['records'], right['records']):
        assert local['source_metadata'] == native['source_metadata']
        assert local['received']['wire_sha256'] == native['received']['wire_sha256']
        assert local['checkpoint_sha256'] == native['checkpoint_sha256']
        assert local['received']['readonly_states'] == native['received']['readonly_states']
        for camera in ('left', 'right'):
            different += local['received']['arrays'][camera]['sha256'] != native['received']['arrays'][camera]['sha256']
    result = dict(state='passed_all12_transferred_trained_float_reception_view_records', checked_unix=time.time(),
                  reports={str(p.relative_to(ROOT)): sha(p) for p in paths}, sources=left['sources'],
                  training_sources=left['training_sources'], protocol_sha256=left['protocol_sha256'],
                  checkpoint_sha256=selected, families_each=6, views_each=6, pixels_each=27648,
                  maximum_RGB_errors_each=[report['maximum_RGB_error'] for report in reports],
                  maximum_model_errors_each=[report['maximum_model_error'] for report in reports],
                  cross_host_wire_hashes_equal=True, cross_host_weight_hashes_equal=True,
                  cross_host_output_hashes_different_views=different,
                  limitation='Complete CPU report identity verification; independent calculations ran on each host. No cross-host output equality, native full KITTI receiver or AP.')
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({key: result[key] for key in ('state', 'maximum_RGB_errors_each', 'cross_host_output_hashes_different_views')}))


if __name__ == '__main__':
    main()
