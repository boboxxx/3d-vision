"""Fresh postdecoder process: received header and this packet's neural output only."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

from contract import codec, need, read, save, sha, crop_arrays, describe, input_guard


def run(container, directory):
    container = container.resolve(); directory = directory.resolve()
    output = directory / 'received-RGB-FP32.npz'; report_path = directory / 'crop.json'
    need(not output.exists() and not report_path.exists(), 'unique crop outputs')
    decoded_path = directory / 'reconstructed.npz'
    sys.addaudithook(input_guard([decoded_path]))
    c = codec(); blob = container.read_bytes(); parsed = c.unpack(blob, c.tables()[1])
    report = read(directory / 'receiver.json')
    need(report['state'] == 'passed' and report['container_sha256'] == hashlib.sha256(blob).hexdigest(), 'actual same-packet decoder report')
    need(report['public_dimensions'] == {k: parsed[k] for k in ('original_hw', 'padded_hw')}, 'header-bound crop dimensions')
    need(report['calls'] == dict(E=0, HE=0, HD=1, D=1) and report['source_NPZ_read_barrier'], 'causal source-free receiver')
    need(sha(decoded_path) == report['output_sha256'], 'same receiver-produced arrays')
    with np.load(decoded_path, allow_pickle=False) as values:
        arrays = crop_arrays(values['pred_left'], values['pred_right'], parsed['original_hw'], parsed['padded_hw'])
    buffer = io.BytesIO(); np.savez(buffer, **arrays); raw = buffer.getvalue(); output.write_bytes(raw)
    record = dict(state='passed', container_sha256=hashlib.sha256(blob).hexdigest(), receiver_report_sha256=sha(directory / 'receiver.json'),
                  reconstructed_sha256=report['output_sha256'], original_hw=parsed['original_hw'], padded_hw=parsed['padded_hw'],
                  crop='top_left_public_header_NCHW[:, :, :h, :w]', arrays={k: describe(v) for k, v in arrays.items()},
                  output_sha256=hashlib.sha256(raw).hexdigest(), unclipped_FP32=True, no_resize_or_RGB8_conversion=True,
                  source_NPZ_read_barrier=True, allowed_input_NPZ=str(decoded_path),
                  receiver_inputs=['actual_received_P6EC_public_header', 'this_packet_neural_reconstructed_arrays', 'this_packet_receiver_report'])
    save(report_path, record)
    print(json.dumps(dict(state='passed', original_hw=parsed['original_hw'], cache_sha256=record['output_sha256'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--container', type=Path, required=True); p.add_argument('--directory', type=Path, required=True)
    args = p.parse_args(); run(args.container, args.directory)
