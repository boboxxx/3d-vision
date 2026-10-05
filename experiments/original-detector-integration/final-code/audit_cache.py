"""Independent lossless array/resource/noise/source audit of all372 received frames."""
import argparse
import math
from pathlib import Path
import time

import numpy as np
import torch
from common import (ROOT, FINAL_PROTOCOL_SHA, check, current_sources, final_original_chain,
                    load_received, noise_identity, ordered_ids, read, save, sha256)


def expected_resources(row):
    # Independently express9-channel geometry/ROI-union and7-repeatedCRC controls.
    h, w = row['views'][0]['shape']
    hp, wp = h + (-h) % 6, w + (-w) % 6
    cells = []
    for view in row['views']:
        check(view['shape'] == [h, w], 'stereo source shape')
        support = np.zeros((hp // 2, wp // 2), dtype=bool)
        for box in view['boxes']:
            x1, y1, x2, y2 = box['xyxy']
            support[y1 // 2:(y2 + 1) // 2, x1 // 2:(x2 + 1) // 2] = True
        cells.append(int(support.sum()))
    real = 9 * (2 * (hp // 6) * (wp // 6) + sum(cells))
    data = (real + 1) // 2
    control = 1176 + 672 * sum(len(view['boxes']) for view in row['views'])
    return dict(key_cells=cells, data_real_values=real, padding_real_values=real % 2,
                data_uses=data, control_uses=control, pilot_uses=0, total_uses=data + control,
                cbr_complex_per_rgb_real_value=(data + control) / (6 * h * w))


def audit_frame_resources(actual, roi):
    expected = expected_resources(roi)
    check(set(actual) == set(expected) | {'total_energy'}, 'physical accounting schema')
    for key, value in expected.items():
        check(actual[key] == value, 'independent physical formula: ' + key)
    check(math.isfinite(actual['total_energy']) and
          math.isclose(actual['total_energy'], expected['total_uses'], rel_tol=1e-4, abs_tol=1e-4),
          'full unit-energy data/control channel use')
    return expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    check(not args.output.exists(), 'retain previous cache audit')
    record = read(args.manifest)
    ids = ordered_ids()
    check(record['state'] == 'finished' and record['ordered_ids'] == ids and record['frames_complete'] == 372
          and set(record['frames']) == set(ids), 'full completed cache')
    check(record['seed'] == 17 and record['snr_db'] == 10. and record['device'] == 'cpu'
          and record['protocol_sha256'] == FINAL_PROTOCOL_SHA, 'prelocked condition')
    check(record['codec_states_readonly'] == 768 and len(record['codec_initial_state_hashes']) == 768
          and record['codec_initial_state_hashes'] == record['codec_final_state_hashes'], 'all768 readonly')
    chain = final_original_chain()
    check(record['final_chain'] == chain and record['final_checkpoint_sha256'] == chain['checkpoint_sha256']
          and record['source_identities'] == current_sources(), 'final full chain/source identity')
    rows = [__import__('json').loads(line) for line in
            (ROOT / 'data/engineering/cao2025-roi-holdout-001.jsonl').read_text().splitlines()]
    roi_audit = read(ROOT / 'data/engineering/cao2025-roi-holdout-audit-001.json')
    check([r['frame_id'] for r in rows] == ids and roi_audit['state'] == 'passed'
          and roi_audit['records_sha256'] == sha256(ROOT / 'data/engineering/cao2025-roi-holdout-001.jsonl'),
          'audited ROI sensor provenance')
    generator = torch.Generator().manual_seed(17)
    check(record['generator_initial_state_sha256'] == noise_identity(generator), 'dedicated seed17')
    files = {}
    total_uses = total_energy = erased = 0
    for frame_id, roi in zip(ids, rows):
        outputs, row = load_received(args.manifest, frame_id)
        check(row['frame_id'] == frame_id and row['native_shape'] == [1, 3, *roi['views'][0]['shape']], 'native geometry')
        check(row['sensor_sha256'] == {v['camera']: v['image_sha256'] for v in roi['views']}, 'exact sensor source')
        for camera, digest in row['sensor_sha256'].items():
            check(sha256(Path('/mnt/d/paper6/data/kitti/training') / camera / f'{frame_id}.png') == digest, 'fresh sensor file')
        expected = audit_frame_resources(row['accounting'], roi)
        check(row['receiver_calls'] == dict(receive_payload=1, semantic_decode=int(row['erasure'] is None)), 'received-only actual API')
        check(row['generator_before_state_sha256'] == noise_identity(generator), 'independent before-noise state')
        if record['channel'] == 'awgn':
            torch.randn((expected['total_uses'], 2), dtype=torch.float32, generator=generator)
        else:
            check(record['channel'] == 'identity' and row['erasure'] is None, 'identity cannot have radio erasure')
        check(row['generator_after_state_sha256'] == noise_identity(generator), 'independent actual noise draw replay')
        if outputs is not None:
            check(row['decoder_range'] == [{k: v for k, v in item.items() if k in
                                           ('minimum', 'maximum', 'out_of_range_fraction')}
                                          for item in row['received_tensors']], 'unclipped received range')
        else:
            check(row['decoder_range'] is None and isinstance(row['erasure'], str), 'true erasure scope')
            erased += 1
        files[frame_id] = dict(received_file_sha256=row['received_file_sha256'], received_tensors=row['received_tensors'])
        total_uses += expected['total_uses']
        total_energy += row['accounting']['total_energy']
    check(record['generator_final_state_sha256'] == noise_identity(generator), 'independent final-noise state')
    check(current_sources() == record['source_identities'], 'source changed during audit')
    result = dict(state='passed', scope='full372 native received-array/resource/source audit, no AP',
                  channel=record['channel'], frames=372, erasures=erased,
                  mean_complex_uses=total_uses / 372, attempted_complex_uses=total_uses,
                  attempted_energy=total_energy, manifest_sha256=sha256(args.manifest),
                  checkpoint_sha256=chain['checkpoint_sha256'], files=files, checked_at_unix=time.time())
    save(args.output, result)
    print(__import__('json').dumps({k: v for k, v in result.items() if k != 'files'}))


if __name__ == '__main__':
    main()
