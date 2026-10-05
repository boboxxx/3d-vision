"""Unique fixed-budget train-only native SRCNN adaptation; no AP selection."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch

import common as c
from model import central_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', choices=('engineering', 'formal'), required=True)
    parser.add_argument('--rate', type=int, choices=(10, 30, 50), required=True)
    args = parser.parse_args()
    prefix = f'srcnn-kitti-{args.scope}-seed17-002-cr{args.rate}'
    directory = c.NATIVE / prefix
    manifest = c.ROOT / 'data/runs' / (prefix + '.json')
    assert not directory.exists() and not manifest.exists()
    sources = c.sources()
    local = c.read(c.ROOT / 'data/engineering/srcnn-KITTI-adaptation-local-CPU-002.json')
    native = c.read(c.ROOT / 'data/engineering/srcnn-KITTI-adaptation-sheng-CPU-002.json')
    assert local['state'] == native['state'] == 'passed_five_SRCNN_training_CPU_families'
    assert local['sources'] == native['sources'] == sources
    assert local['author_initial_float32_states'] == native['author_initial_float32_states']
    if args.scope == 'formal':
        engineering = c.read(c.ROOT / 'data/provenance/srcnn-kitti-engineering-seed17-002-closure.json')
        transferred = c.read(c.ROOT / 'data/provenance/srcnn-kitti-engineering-seed17-002-local-verification.json')
        assert engineering['actual_terminal'] and engineering['sources'] == sources
        assert transferred['closure_sha256'] == c.sha(c.ROOT / 'data/provenance/srcnn-kitti-engineering-seed17-002-closure.json')
        assert transferred['state'] == 'passed_all18_transferred_SRCNN_training_records_and_terminal_artifacts'
        main = c.read(c.ROOT / 'data/provenance/source-cache-inference-main-001-closure.json')
        main_local = c.read(c.ROOT / 'data/provenance/source-cache-inference-main-001-local-verification.json')
        assert main['actual_terminal'] and main['frames'] == 45228
        assert main_local['state'] == 'passed_all45228_transferred_native_inference_records_predictions_and_metrics'
        assert main_local['closure_sha256'] == c.sha(c.ROOT / 'data/provenance/source-cache-inference-main-001-closure.json')
    free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip()) * 2**20
    assert free >= 12 * 2**30 and os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    torch.set_num_threads(2); torch.manual_seed(17); torch.cuda.manual_seed_all(17)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    frame_ids = c.ids(args.scope)
    epochs = 6 if args.scope == 'engineering' else 20
    updates = 6 if args.scope == 'engineering' else 37120
    allowed_images = {str((c.DATA / 'training' / camera / (frame + '.png')).resolve())
                      for frame in frame_ids for camera in ('image_2', 'image_3')}
    def guard(event, values):
        if event != 'open' or not isinstance(values[0], (str, bytes)):
            return
        name = os.fsdecode(values[0]); mode, flags = values[1:3]
        reading = (isinstance(mode, str) and 'r' in mode) or (mode is None and flags & os.O_ACCMODE == os.O_RDONLY)
        if not reading:
            return
        assert '/label_2/' not in name and '/calib/' not in name and '/velodyne/' not in name
        assert not name.lower().endswith(('.pkl', '.npz', '.pth', '.pt', '.bin'))
        if name.lower().endswith('.png'):
            assert str(Path(name).resolve()) in allowed_images
        if name.lower().endswith('.mat'):
            assert Path(name).resolve() == c.MODEL.resolve()
    # Initialize torch/optimizer before installing the actual data-read barrier.
    model = c.initial_model().cuda().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=0)
    assert c.states(model) == local['author_initial_float32_states']
    sys.addaudithook(guard)
    directory.mkdir()
    raw = directory / 'train.jsonl'
    run = dict(state='running', pid=os.getpid(), scope=args.scope, rate=args.rate, epochs=epochs,
               required_updates=updates, optimizer_steps=0, ordered_ids=frame_ids, sources=sources,
               source_model_sha256=c.sha(c.MODEL), initial_state=c.states(model), trainable_tensors=6,
               data_barrier=True, no_GT_calibration_detector_or_validation=True, actual_source_images_sha256={},
               started_unix=time.time(), directory=str(directory), raw_training_path=str(raw),
               physical_free_before_bytes=free, TF32=False, mixed_precision=False, batch_workers=0, CPU_threads=2)
    c.save(manifest, run)
    generator = np.random.Generator(np.random.PCG64(1719))
    torch.cuda.reset_peak_memory_stats()
    try:
        with raw.open('x', buffering=1) as stream:
            for epoch, step, views in c.order(frame_ids, epochs):
                images, view_records = [], []
                for frame, camera in views:
                    path = c.DATA / 'training' / camera / (frame + '.png')
                    digest = c.sha(path)
                    assert str(path) not in run['actual_source_images_sha256'] or run['actual_source_images_sha256'][str(path)] == digest
                    run['actual_source_images_sha256'][str(path)] = digest
                    with Image.open(path) as image:
                        assert image.mode == 'RGB'
                        images.append(np.array(image, dtype=np.uint8))
                    view_records.append(dict(frame_id=frame, camera=camera, source_path=str(path), source_sha256=digest))
                inputs, targets, sampling = c.patches(images, args.rate, generator)
                input_identity, target_identity = c.describe(inputs), c.describe(targets)
                inputs, targets = inputs.cuda(), targets.cuda()
                optimizer.zero_grad(set_to_none=True)
                loss = central_loss(model(inputs), targets)
                assert torch.isfinite(loss)
                loss.backward()
                gradients = {}
                for name, parameter in model.named_parameters():
                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                    gradients[name] = dict(norm=float(parameter.grad.norm()), nonzero=int(torch.count_nonzero(parameter.grad)))
                optimizer.step()
                adam_steps = {}
                for name, parameter in model.named_parameters():
                    assert torch.isfinite(parameter).all()
                    state = optimizer.state[parameter]
                    assert set(state) == {'step', 'exp_avg', 'exp_avg_sq'}
                    assert int(state['step']) == step and torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all()
                    adam_steps[name] = int(state['step'])
                assert torch.cuda.max_memory_reserved() + 2 * 2**30 <= free
                row = dict(step=step, epoch=epoch, rate=args.rate, loss=float(loss.detach()), views=view_records,
                           sampling=sampling, input_tensor=input_identity, target_tensor=target_identity,
                           gradients=gradients, parameter_tensors=6, adam_steps=adam_steps,
                           peak_reserved_bytes=torch.cuda.max_memory_reserved())
                stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
                run['optimizer_steps'] = step
                if step % 100 == 0:
                    c.save(manifest, run)
                    print(json.dumps(dict(rate=args.rate, epoch=epoch, updates=step, loss=row['loss'])), flush=True)
                del images, inputs, targets, loss
        assert run['optimizer_steps'] == updates and len(run['actual_source_images_sha256']) == len(allowed_images)
        assert c.sources() == sources
        assert all(c.sha(path) == digest for path, digest in run['actual_source_images_sha256'].items())
        checkpoint = directory / 'final.pth'
        buffer = io.BytesIO()
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), optimizer_steps=updates,
                        rate=args.rate, sources=sources, initial_state=run['initial_state'], epochs=epochs), buffer)
        serialized = buffer.getvalue()
        checkpoint.write_bytes(serialized)
        run.update(state='finished', ended_unix=time.time(), final_state=c.states(model),
                   checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(serialized).hexdigest(), raw_sha256=c.sha(raw),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(), selected_epoch=epochs,
                   limiter='Training-only declared adaptation; no main reconstruction, SRCNN AP, radio or novelty result')
        c.save(manifest, run)
    except BaseException as error:
        run.update(state='failed', error=repr(error), ended_unix=time.time()); c.save(manifest, run); raise
    print(json.dumps(dict(state=run['state'], rate=args.rate, updates=updates)), flush=True)


if __name__ == '__main__':
    main()
