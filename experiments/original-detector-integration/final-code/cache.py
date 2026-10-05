"""Generate two lossless received RGB caches from the sole final audited codec."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys
import time

import torch

from common import (ROOT, HERE, FINAL_PROTOCOL_SHA, check, current_sources,
                    final_original_chain, noise_identity, ordered_ids, save, sha256,
                    tensor_identity)
sys.path[:0] = [str(HERE.parent), str(ROOT / 'reproduction/cao2025')]
from adapter import transmit_rgb
from data import StereoRGB
from wireless import WirelessVariant
import radio
from geocomm.pooling_diagnostic import state_hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    check(re.fullmatch(r'[a-z0-9][a-z0-9-]{3,90}', args.prefix), 'safe unique prefix')
    chain = final_original_chain()
    sources = current_sources()
    torch.set_num_threads(2)
    torch.manual_seed(17)
    spec = importlib.util.spec_from_file_location('paper6_final_cao_semantic', ROOT / 'reproduction/cao2025/model.py')
    semantic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(semantic)
    model = WirelessVariant(semantic.SemanticVariant('/mnt/d/paper6/checkpoints/spynet_sintel_final-3d2a1287.pth')).cpu().eval()
    state = torch.load(chain['checkpoint_path'], map_location='cpu', weights_only=False)['model_state']
    actual = model.state_dict()
    check(len(actual) == len(state) == 768 and set(actual) == set(state), 'complete768 final state')
    check(all(v.dtype == actual[k].dtype and v.shape == actual[k].shape and torch.isfinite(v).all()
              for k, v in state.items()), 'complete final dtype/shape/finite')
    model.load_state_dict(state, strict=True)
    check(all(torch.equal(v, state[k]) for k, v in model.state_dict().items()), 'strict final values')
    del state, actual
    before = state_hashes(model)

    def forbidden(*args, **kwargs):
        raise RuntimeError('broad training forward or clean semantic reconstruction called')

    model.forward = forbidden
    model.semantic_mse = forbidden
    receive, decode = model.receive_payload, model.semantic.decode
    calls = {}
    pending = {}

    def receive_only(observation):
        check(isinstance(observation, radio.Observation) and observation.symbols.device.type == 'cpu'
              and set(vars(observation)) == {'symbols', 'noise_power', 'channel'}, 'received observation only')
        calls['receive_payload'] += 1
        result = receive(observation)
        pending['received'] = result
        return result

    def decode_only(received):
        check(received is pending['received'], 'semantic decoder must receive actual receiver payload')
        calls['semantic_decode'] += 1
        return decode(received)

    model.receive_payload = receive_only
    model.semantic.decode = decode_only
    dataset = StereoRGB('/mnt/d/paper6/data/kitti', ROOT / 'data/engineering/cao2025-roi-holdout-001.jsonl',
                        ROOT / 'data/engineering/cao2025-roi-holdout-audit-001.json',
                        ROOT / 'data/internal-tuning-fold-001.json', 'geocomm_tune_holdout')
    ids = ordered_ids()
    roi_rows = dataset.rows
    for channel in ('identity', 'awgn'):
        output = Path('/mnt/d/paper6/runs') / f'{args.prefix}-{channel}'
        manifest = ROOT / f'data/runs/{args.prefix}-{channel}.json'
        check(not output.exists() and not manifest.exists(), 'retain all previous cache outputs')
        output.mkdir(parents=True)
        generator = torch.Generator().manual_seed(17)
        record = dict(state='running', channel=channel, snr_db=10., seed=17, device='cpu',
                      output_dir=str(output), ordered_ids=ids, frames_complete=0, frames={},
                      source_identities=sources, final_chain=chain, final_checkpoint_sha256=chain['checkpoint_sha256'],
                      protocol_sha256=FINAL_PROTOCOL_SHA, codec_initial_state_hashes=before,
                      started_at_unix=time.time(), generator_initial_state_sha256=noise_identity(generator))
        save(manifest, record)
        try:
            with torch.no_grad():
                for index, frame_id in enumerate(ids):
                    frame = dataset[index]
                    check(frame['frame_id'] == frame_id, 'native fixed order')
                    check(all(not m.training for m in model.modules()), 'all codec modules eval')
                    calls.update(receive_payload=0, semantic_decode=0)
                    pending.clear()
                    rng_before = noise_identity(generator)
                    global_rng = torch.get_rng_state().clone()
                    result = transmit_rgb(model, radio, frame['left'], frame['right'], frame['boxes'],
                                          channel, 10., generator)
                    expected = dict(receive_payload=1, semantic_decode=int(result['erasure'] is None))
                    check(calls == expected, 'actual receiver API scope')
                    check(torch.equal(global_rng, torch.get_rng_state()), 'radio inference consumed global RNG')
                    path = output / f'{frame_id}.pth'
                    tensors = result['outputs']
                    torch.save(dict(frame_id=frame_id, channel=channel, outputs=tensors), path)
                    row = dict(frame_id=frame_id, native_shape=list(frame['left'].shape),
                               sensor_sha256={v['camera']: v['image_sha256'] for v in roi_rows[index]['views']},
                               received_path=str(path), received_file_sha256=sha256(path),
                               received_tensors=[tensor_identity(v) for v in tensors] if tensors is not None else None,
                               decoder_range=result['decoder_range'], accounting=result['accounting'],
                               erasure=result['erasure'], receiver_calls=dict(calls),
                               generator_before_state_sha256=rng_before,
                               generator_after_state_sha256=noise_identity(generator))
                    check(state_hashes(model) == before, 'full768 codec state mutated')
                    record['frames'][frame_id] = row
                    record['frames_complete'] = index + 1
                    save(manifest, record)
                    if (index + 1) % 20 == 0:
                        print(json.dumps(dict(channel=channel, frames=index + 1)), flush=True)
                    del result, tensors, frame
                    pending.clear()
            check(current_sources() == sources and sha256(chain['checkpoint_path']) == chain['checkpoint_sha256'],
                  'sources/final checkpoint changed')
            record.update(state='finished', codec_states_readonly=768, codec_final_state_hashes=state_hashes(model),
                          generator_final_state_sha256=noise_identity(generator), ended_at_unix=time.time(),
                          limitations='Declared final original variant; internal372 with author-pretraining overlap, unmatched rates/exposures; no AP or fair gain from cache alone')
            save(manifest, record)
        except BaseException as exc:
            record.update(state='failed', exception=repr(exc), ended_at_unix=time.time())
            save(manifest, record)
            raise


if __name__ == '__main__':
    main()
