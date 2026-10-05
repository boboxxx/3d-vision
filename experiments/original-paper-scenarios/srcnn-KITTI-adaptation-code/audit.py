"""Fresh all-update sampler/PNG/patch/Adam/final-weight audit, no model replay."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

import common as c


def independent_patches(images, rate, sampling):
    inputs, targets = [], []
    matrix = np.array([[65.481, 128.553, 24.966], [-37.797, -74.203, 112.0],
                       [112.0, -93.786, -18.214]], dtype=np.float64)
    offset = np.array([16.0, 128.0, 128.0], dtype=np.float64)
    for rgb, item in zip(images, sampling):
        h, w = rgb.shape[:2]
        lh, lw = int(math.floor(h / math.sqrt(rate))), int(math.floor(w / math.sqrt(rate)))
        low = np.asarray(Image.fromarray(rgb).resize((lw, lh), Image.Resampling.BICUBIC), dtype=np.uint8)
        restored = []
        for channel in range(3):
            plane = low[:, :, channel].astype(np.float32) / np.float32(255)
            restored.append(np.asarray(Image.fromarray(plane).resize((w, h), Image.Resampling.BICUBIC), dtype=np.float64))
        colors = np.stack(restored, axis=-1)
        y = ((colors @ matrix.T + offset) / 255)[:, :, 0].astype(np.float32)
        clean = ((rgb.astype(np.float64) / 255 @ matrix.T + offset) / 255)[:, :, 0].astype(np.float32)
        assert item['native_hw'] == [h, w] and item['native_RGB8'] == c.describe(rgb)
        for top, left in item['patch_coordinates']:
            inputs.append(y[top:top + 49, left:left + 49])
            targets.append(clean[top:top + 49, left:left + 49])
    return c.describe(np.stack(inputs)[:, None]), c.describe(np.stack(targets)[:, None])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); assert not args.output.exists()
    torch.set_num_threads(2)
    run = c.read(args.manifest)
    assert run['state'] == 'finished' and run['sources'] == c.sources()
    assert run['ordered_ids'] == c.ids(run['scope'])
    assert run['data_barrier'] and run['no_GT_calibration_detector_or_validation']
    epochs, updates = (6, 6) if run['scope'] == 'engineering' else (20, 37120)
    assert run['epochs'] == run['selected_epoch'] == epochs
    assert run['required_updates'] == run['optimizer_steps'] == updates
    assert not run['TF32'] and not run['mixed_precision'] and run['batch_workers'] == 0
    assert run['peak_reserved_bytes'] + 2 * 2**30 <= run['physical_free_before_bytes']
    assert c.sha(run['checkpoint']) == run['checkpoint_sha256'] and c.sha(run['raw_training_path']) == run['raw_sha256']
    initial = c.states(c.initial_model())
    assert initial == run['initial_state'] and len(initial) == run['trainable_tensors'] == 6
    checkpoint = torch.load(run['checkpoint'], map_location='cpu', weights_only=False)
    assert checkpoint['sources'] == run['sources'] and checkpoint['initial_state'] == initial
    assert checkpoint['optimizer_steps'] == updates and checkpoint['rate'] == run['rate'] and checkpoint['epochs'] == epochs
    assert {name: c.describe(value) for name, value in checkpoint['model'].items()} == run['final_state']
    assert set(checkpoint['model']) == set(initial) and run['final_state'] != initial
    optimizer = checkpoint['optimizer']
    assert len(optimizer['param_groups']) == 1 and set(optimizer['state']) == set(range(6))
    group = optimizer['param_groups'][0]
    assert group['params'] == list(range(6)) and group['lr'] == 1e-4
    assert group['betas'] == (.9, .999) and group['eps'] == 1e-8 and group['weight_decay'] == 0
    for index, parameter in enumerate(checkpoint['model'].values()):
        state = optimizer['state'][index]
        assert set(state) == {'step', 'exp_avg', 'exp_avg_sq'} and int(state['step']) == updates
        assert state['exp_avg'].shape == state['exp_avg_sq'].shape == parameter.shape
        assert torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all() and torch.isfinite(parameter).all()
    ids = run['ordered_ids']
    views = [(frame, camera) for frame in ids for camera in ('image_2', 'image_3')]
    expected_pngs = {str(c.DATA / 'training' / camera / (frame + '.png')) for frame, camera in views}
    assert set(run['actual_source_images_sha256']) == expected_pngs
    # Independent generator/order reduction; do not call trainer order/patch helpers.
    order_rng = np.random.Generator(np.random.PCG64(17))
    patch_rng = np.random.Generator(np.random.PCG64(1719))
    rows_checked = 0
    target_hashes = []
    with Path(run['raw_training_path']).open() as stream:
        for epoch in range(1, epochs + 1):
            permutation = order_rng.permutation(len(views))
            for offset in range(0, len(views), 4):
                line = stream.readline(); assert line.endswith('\n')
                row = json.loads(line); rows_checked += 1
                assert row['step'] == rows_checked and row['epoch'] == epoch and row['rate'] == run['rate']
                expected_views = [views[int(index)] for index in permutation[offset:offset + 4]]
                assert [(item['frame_id'], item['camera']) for item in row['views']] == expected_views
                assert row['parameter_tensors'] == 6 and set(row['gradients']) == set(row['adam_steps']) == set(initial)
                assert set(row['adam_steps'].values()) == {rows_checked}
                assert math.isfinite(row['loss']) and row['loss'] >= 0
                assert all(math.isfinite(value['norm']) and value['norm'] >= 0 and value['nonzero'] >= 0
                           for value in row['gradients'].values())
                assert row['peak_reserved_bytes'] <= run['peak_reserved_bytes']
                images = []
                assert len(row['sampling']) == len(row['views']) == 4
                for item, sample in zip(row['views'], row['sampling']):
                    path = c.DATA / 'training' / item['camera'] / (item['frame_id'] + '.png')
                    assert str(path) == item['source_path'] and c.sha(path) == item['source_sha256'] == run['actual_source_images_sha256'][str(path)]
                    with Image.open(path) as image:
                        assert image.mode == 'RGB'
                        rgb = np.array(image, dtype=np.uint8)
                    h, w = rgb.shape[:2]
                    expected_coordinates = [[int(patch_rng.integers(0, h - 49 + 1)), int(patch_rng.integers(0, w - 49 + 1))]
                                            for _ in range(8)]
                    assert sample['patch_coordinates'] == expected_coordinates
                    images.append(rgb)
                inputs, targets = independent_patches(images, run['rate'], row['sampling'])
                assert inputs == row['input_tensor'] and targets == row['target_tensor']
                target_hashes.append(targets['sha256'])
        assert stream.read() == ''
    assert rows_checked == updates and c.sources() == run['sources']
    result = dict(state='passed_all_updates_native_independent_patches_and_final_Adam', checked_unix=time.time(),
                  scope=run['scope'], rate=run['rate'], updates=updates, patches=updates * 32,
                  unique_source_views=len(views), source_view_visits=updates * 4,
                  manifest_sha256=c.sha(args.manifest), raw_sha256=run['raw_sha256'], checkpoint_sha256=run['checkpoint_sha256'],
                  sources=run['sources'], target_hashes_by_step=target_hashes,
                  final_state=run['final_state'], readonly_author_model_sha256=c.sha(c.MODEL),
                  limitation='Fresh actual PNG/patch/optimizer/all-update checks, no model/backpropagation replay or AP')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(state=result['state'], rate=run['rate'], updates=updates)), flush=True)


if __name__ == '__main__':
    main()
