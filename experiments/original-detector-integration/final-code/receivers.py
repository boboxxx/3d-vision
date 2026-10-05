"""Native received-image-only author detectors; no codec or inference labels."""
import importlib
import sys
import tempfile

import numpy as np
import torch
import torch.distributed as dist

from common import ROOT, HERE, check, sha256
sys.path.insert(0, str(HERE.parent))
from adapter import liga_preprocess, stereo_rcnn_preprocess
from geocomm.pooling_diagnostic import state_hashes


class StereoReceiver:
    name = 'stereo_rcnn'
    states = 670
    checkpoint_sha = 'b7a07a897224f75bc30b2d4c06f39927e92933a8bcaad6e679aff605b2346c14'

    def __init__(self):
        stereo = ROOT / 'third_party/Stereo-RCNN'
        sys.path.insert(0, str(stereo / 'lib'))
        from model.stereo_rcnn.resnet import resnet
        from model.utils.config import cfg
        from model.utils.blob import prep_im_for_blob
        from model.utils.kitti_utils import read_obj_calibration
        from model.dense_align import dense_align
        from geocomm.stereo_baseline import decode_3d, kitti_line
        self.cfg, self.prep, self.read_calib = cfg, prep_im_for_blob, read_obj_calibration
        self.decode_3d, self.kitti_line = decode_3d, kitti_line
        binaries = list((stereo / 'lib/model').glob('_C*.so'))
        check(len(binaries) == 1 and sha256(binaries[0]) ==
              '5c3cbc235cad8e67dbb440cb01940d912f72b702455e51dafde2573f7d9c6429', 'author native operator')
        self.model = resnet(np.asarray(['__background__', 'Car']), 101, pretrained=False)
        self.model.create_architecture()
        path = '/mnt/d/paper6/checkpoints/stereo-rcnn-branch1-author.pth'
        check(sha256(path) == self.checkpoint_sha, 'author670 checkpoint')
        state = torch.load(path, map_location='cpu', weights_only=False)['model']
        actual = self.model.state_dict()
        check(len(state) == len(actual) == self.states and set(state) == set(actual), 'complete670 states')
        check(all(v.dtype == actual[k].dtype and v.shape == actual[k].shape and torch.isfinite(v).all()
                  for k, v in state.items()), 'strict670 dtype/shape/finite')
        self.model.load_state_dict(state, strict=True)
        check(all(torch.equal(v, state[k]) for k, v in self.model.state_dict().items()), 'full670 values')
        del state, actual
        self.model.cuda().eval()
        self.initial = state_hashes(self.model)
        self.calls, self.pending, self.handles = {}, {}, []

        def forbidden(*args):
            raise RuntimeError('native proposal-target training call forbidden')

        def entry(module, args):
            check(not module.training and not torch.is_grad_enabled(), 'native eval/no_grad')
            check(args[0] is self.pending['images'][0] and args[1] is self.pending['images'][1], 'received native entry')
            check(all(torch.count_nonzero(value) == 0 for value in args[3:]), 'zero unused GT placeholders only')
            self.calls['detector'] += 1

        def backbone(module, args):
            check(args[0] is self.pending['images'][self.calls['image_backbone']], 'received native backbone')
            self.calls['image_backbone'] += 1

        self.original_align = dense_align.align_parallel
        self.dense = dense_align

        def alignment(*args, **kwargs):
            check(not torch.is_grad_enabled() and args[2] is self.pending['images'][0]
                  and args[3] is self.pending['images'][1], 'received dense alignment')
            self.calls['dense_alignment'] += 1
            return self.original_align(*args, **kwargs)

        self.handles.extend([self.model.register_forward_pre_hook(entry),
                             self.model.RCNN_layer0.register_forward_pre_hook(backbone),
                             self.model.RCNN_proposal_target.register_forward_pre_hook(forbidden)])
        dense_align.align_parallel = alignment

    def predict(self, outputs, frame_id, calibration_path, directory):
        self.calls.update(detector=0, image_backbone=0, dense_alignment=0)
        self.pending.clear()
        images, info, shape = stereo_rcnn_preprocess(outputs, self.prep, self.cfg.PIXEL_MEANS,
                                                    self.cfg.TRAIN.SCALES[0], self.cfg.TRAIN.MAX_SIZE)
        tensors = tuple(v.cuda() for v in images)
        native_info = info.cuda()
        self.pending['images'] = tensors
        calibration = self.read_calib(str(calibration_path))
        gt = torch.zeros(1, 5, device='cuda')
        count = torch.zeros(1, dtype=torch.long, device='cuda')
        result = self.model(*tensors, native_info, gt, gt, gt, gt, gt, count)
        predictions, counts = self.decode_3d(result, *tensors, native_info, shape, calibration, threshold=.05)
        lines = [self.kitti_line(row, calibration) for row in predictions]
        check(self.calls == dict(detector=1, image_backbone=2,
                                  dense_alignment=int(counts['initial_solutions'] > 0)), 'actual native calls')
        (directory / f'{frame_id}.txt').write_text(''.join(lines))
        self.pending.clear()
        return dict(calls=dict(self.calls), prediction_count=len(predictions), native_3D_counts=counts,
                    processed_shape=list(tensors[0].shape), original_shape=list(shape), GT_placeholder_only=True)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.dense.align_parallel = self.original_align


class LigaReceiver:
    name = 'liga'
    states = 484
    checkpoint_sha = '3f36af30b686f23d0868482675177b29cc50c2b378577dd12f9092ddb0235b2e'

    def __init__(self):
        import os
        liga = ROOT / 'third_party/LIGA-Stereo'
        sys.path[:0] = [str(liga), str(ROOT / 'third_party/mmdetection_kitti')]
        os.chdir(liga)
        from easydict import EasyDict
        from liga.config import cfg_from_yaml_file
        from liga.datasets import build_dataloader
        from liga.datasets.augmentor.stereo_data_augmentor import StereoDataAugmentor
        from liga.datasets.stereo_dataset_template import StereoDatasetTemplate
        from liga.models import build_network
        from liga.utils.calibration_kitti import Calibration
        from liga.utils.common_utils import create_logger
        from geocomm.compat import adapt_spconv_state
        from geocomm.inference import SENSOR_INPUT_KEYS
        self.sensor_keys, self.Calibration = SENSOR_INPUT_KEYS, Calibration
        self.collate = StereoDatasetTemplate.collate_batch
        self.cfg = cfg_from_yaml_file(str(ROOT / 'configs/diagnostic/clean_holdout.yaml'), EasyDict())
        self.dataset, _, _ = build_dataloader(self.cfg.DATA_CONFIG, self.cfg.CLASS_NAMES, batch_size=1,
                                             dist=False, workers=0, training=False, logger=create_logger())
        self.augmentor = StereoDataAugmentor(self.dataset.root_path, self.cfg.DATA_CONFIG.TEST_DATA_AUGMENTOR,
                                            self.cfg.CLASS_NAMES)
        self.group = tempfile.TemporaryDirectory(prefix='original-final-LIGA-')
        dist.init_process_group('nccl', init_method='file://' + self.group.name + '/rank', rank=0, world_size=1)
        self.model = build_network(self.cfg.MODEL, len(self.cfg.CLASS_NAMES), self.dataset).cuda().eval()
        b = self.model.backbone_3d
        check(all(v is None for v in (b.semantic_link, b.stereo_feature_link, b.student_semantic_link_encoder,
                                      self.model.rgb_semantic_link)), 'all added links/student disabled')
        path = '/mnt/d/paper6/checkpoints/liga-author-download'
        check(sha256(path) == self.checkpoint_sha, 'author484 checkpoint')
        state = adapt_spconv_state(self.model, torch.load(path, map_location='cpu', weights_only=False)['model_state'])
        actual = self.model.state_dict()
        check(len(state) == len(actual) == self.states and set(state) == set(actual), 'complete484 states')
        check(all(v.dtype == actual[k].dtype and v.shape == actual[k].shape and torch.isfinite(v).all()
                  for k, v in state.items()), 'strict484 dtype/shape/finite')
        self.model.load_state_dict(state, strict=True)
        check(all(torch.equal(v.detach().cpu(), state[k]) for k, v in self.model.state_dict().items()), 'full484 values')
        del state, actual
        self.initial = state_hashes(self.model)
        # Keep native grad flags; inference uses no_grad and no optimizer.
        self.wrapped = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[0], broadcast_buffers=False)
        self.calls, self.pending, self.handles = {}, {}, []

        def forbidden(*args, **kwargs):
            self.calls['forbidden'] += 1
            raise RuntimeError('GT/LiDAR teacher or training loss called')

        def entry(module, args):
            check(not module.training and not torch.is_grad_enabled() and set(args[0]) == set(self.sensor_keys),
                  'native sensor-only eval entry')
            check(args[0]['left_img'] is self.pending['images'][0] and
                  args[0]['right_img'] is self.pending['images'][1], 'actual received detector entry')

        def backbone(module, args):
            check(args[0] is self.pending['images'][self.calls['image_backbone']], 'actual received image backbone')
            self.calls['image_backbone'] += 1

        def count(name):
            def hook(module, args):
                self.calls[name] += 1
            return hook

        if self.model.lidar_model is not None:
            self.handles.append(self.model.lidar_model.register_forward_pre_hook(forbidden))
        self.model.get_training_loss = forbidden
        self.model.dense_head.get_loss = forbidden
        self.model.depth_loss_head.get_loss = forbidden
        if self.model.dense_head_2d is not None:
            self.model.dense_head_2d.get_loss = forbidden
        self.handles.extend([self.model.register_forward_pre_hook(entry),
                             b.feature_backbone.register_forward_pre_hook(backbone),
                             b.feature_neck.register_forward_pre_hook(count('feature_neck')),
                             b.build_cost.register_forward_pre_hook(count('build_cost')),
                             self.model.dense_head.register_forward_pre_hook(count('head3D'))])

    def predict(self, outputs, frame_id, calibration_path, directory):
        self.calls.update(image_backbone=0, feature_neck=0, build_cost=0, head3D=0, forbidden=0)
        self.pending.clear()
        batch = liga_preprocess(outputs, self.Calibration(str(calibration_path)), frame_id,
                                self.augmentor, self.collate, device='cuda')
        self.pending['images'] = (batch['left_img'], batch['right_img'])
        predictions, _ = self.wrapped({k: batch[k] for k in self.sensor_keys})
        check(self.calls == dict(image_backbone=2, feature_neck=2, build_cost=1, head3D=1, forbidden=0), 'native receiver calls')
        check(all(torch.isfinite(p[k]).all() for p in predictions for k in ('pred_boxes', 'pred_scores')), 'finite predictions')
        self.dataset.generate_prediction_dicts(batch, predictions, self.cfg.CLASS_NAMES, output_path=directory)
        self.pending.clear()
        return dict(calls=dict(self.calls), prediction_count=len(predictions[0]['pred_boxes']),
                    processed_shape=list(batch['left_img'].shape), original_shape=batch['image_shape'].tolist(),
                    crop_offsets=batch['calib'][0].offsets)

    def close(self):
        for handle in self.handles:
            handle.remove()
        if dist.is_initialized():
            dist.destroy_process_group()
        self.group.cleanup()
