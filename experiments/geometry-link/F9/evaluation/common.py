"""F9 inference contracts; sealed coding and native receiver remain unchanged."""
import hashlib
import math
from pathlib import Path
import sys
import types
from contextlib import contextmanager

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'experiments/geometry-link/F9/code')]
from coupling import ARMS, CorrespondenceLink, install
from geocomm.compat import adapt_spconv_state
from geocomm.evidence import sha256, source_identity
from geocomm.pooling_diagnostic import state_hashes
from geocomm.stereo_feature_observer import StereoFeatureObserver
from diagnose_codec_features import compare_features
from geocomm.inference import SENSOR_INPUT_KEYS


def check(condition, message):
    if not condition: raise ValueError(message)


def evaluation_sources():
    return source_identity(ROOT, ['experiments/geometry-link/F9/evaluation'])


def treatment(model, arm, channel, snr):
    check(arm in ARMS and channel in ('identity', 'awgn'), 'public F9 arm/channel')
    check(float(snr) in ((10.,) if channel == 'identity' else (6., 10., 18.)), 'locked public evaluation SNR')
    link = model.backbone_3d.stereo_feature_link
    check(link.channel.kind == channel and link.snr_db == float(snr), 'model/public channel-SNR mismatch')
    install(link, arm)
    return link


def install_native3D_forward(model):
    """Same native 3D modules/postprocessing; auxiliary branches stay loaded/idle."""
    check(not hasattr(model, 'F9_receiver_execution'), 'receiver path installed once')
    def forward(self, sensors):
        check(not self.training and not torch.is_grad_enabled(), 'explicit inference requires eval/no_grad')
        check(set(sensors) <= set(SENSOR_INPUT_KEYS), 'GT/private fields cannot enter 3D receiver inference')
        batch = dict(sensors)
        for module in (self.backbone_3d, self.map_to_bev_module, self.backbone_2d, self.dense_head):
            check(module is not None and not module.training, 'complete eval-only native3D chain')
            batch = module(batch)
        check('gt_boxes' not in batch, 'target inserted during native inference')
        predictions, diagnostics = self.post_processing(batch)
        check(len(predictions) == 1, 'native batch1 prediction')
        for key in ('pred_boxes', 'pred_scores'):
            check(torch.isfinite(predictions[0][key]).all(), 'finite native3D prediction')
        # Preserve the existing DDP-safe accounting carrier for the benchmark writer.
        predictions[0]['batch_dict'] = batch
        return predictions, diagnostics
    model.forward = types.MethodType(forward, model)
    model.F9_receiver_execution = 'explicit_backbone_cost_BEV_native3D_head_postprocessing_no_auxiliary_heads'
    return model


def frozen_load(model, path, expected_sha, loader=None, kwargs=None, role='formal_final'):
    check(sha256(path) == expected_sha, 'fixed complete checkpoint identity')
    loaded = torch.load(path, map_location='cpu', weights_only=False)
    check(isinstance(loaded, dict) and 'model_state' in loaded, 'full state container')
    check(role in ('formal_final', 'engineering_initialization'), 'explicit locked checkpoint role')
    if role == 'formal_final':
        check(loaded.get('arm') == model.backbone_3d.stereo_feature_link.arm and loaded.get('epoch') == 1 and loaded.get('it') == 3340,
              'same-arm sole final3340-step checkpoint')
        check(loaded.get('version') == 'F9-matched-native3D-codec-adaptation', 'formal native checkpoint version')
    else:
        check(loaded.get('version') == 'F9-exact-F8joint-plus-four-head-states' and loaded.get('seed') == 17,
              'zero-update engineering initialization snapshot only')
    state = adapt_spconv_state(model, loaded['model_state'])
    expected = model.state_dict()
    check(len(expected) == len(state) == 539 and set(expected) == set(state), 'exact complete539 states required')
    for name, value in expected.items():
        check(value.shape == state[name].shape and value.dtype == state[name].dtype and torch.isfinite(state[name]).all(), 'state layout/dtype/finite: ' + name)
    if loader is None: model.load_state_dict(state, strict=True)
    else: loader(filename=str(path), **(kwargs or {}))
    check(all(torch.equal(value.detach().cpu(), state[name]) for name, value in model.state_dict().items()), 'native loader failed exact full539 preservation')
    return dict(required_states=539, state_hashes=state_hashes(model), checkpoint_sha256=expected_sha, role=role)


def rng_record(device):
    state = torch.cuda.get_rng_state(device) if device.type == 'cuda' else torch.get_rng_state()
    blob = state.cpu().numpy().tobytes()
    return dict(backend=device.type, device=str(device), state_hex=blob.hex(), sha256=hashlib.sha256(blob).hexdigest())


class F9Observer(StereoFeatureObserver):
    def __init__(self, model, arm, channel, snr, record):
        link = model.backbone_3d.stereo_feature_link
        check(type(link) is CorrespondenceLink and link.arm == arm and link.snr_db == float(snr), 'installed complete F9 treatment')
        self.arm, self.snr = arm, float(snr)
        self.current_device = None
        self.model = model
        self.in_reference = False
        self.reference_calls = dict(dense_head_2d=0, depth_loss_head=0)
        super().__init__(model, channel, record, compare_features)
        self.handles.append(link.channel.register_forward_hook(self.channel_done))
        for name in self.reference_calls:
            module = getattr(model, name)
            if module is not None: self.handles.append(module.register_forward_pre_hook(self.auxiliary))

    def start(self, module, inputs):
        check(not self.in_reference, 'reference cannot make another sender/channel attempt')
        super().start(module, inputs)

    def auxiliary(self, module, inputs):
        if not self.in_reference: return self.forbidden(module, inputs)
        check(not module.training and not torch.is_grad_enabled(), 'reference eval/no_grad')
        check(not any('gt' in k or k == 'random_T' for k in inputs[0]), 'reference cannot access labels/private augmentation')
        name = next(k for k in self.reference_calls if getattr(self.model, k) is module)
        self.reference_calls[name] += 1
        check(self.reference_calls[name] == 1, 'one auxiliary call per engineering condition')

    @contextmanager
    def reference_scope(self):
        check(not self.in_reference and self.pending is None and self.calls['frames'] == 1,
              'reference only after first completed engineering frame')
        check(not any(self.reference_calls.values()), 'one reference per engineering condition')
        check(not self.model.training and not torch.is_grad_enabled(), 'reference eval/no_grad required')
        before = self.calls.copy()
        # The isolated reference may execute only the two auxiliary heads and
        # native 3D head/postprocessing, never sender, teacher or cost construction.
        backbone = self.model.backbone_3d
        handles = [m.register_forward_pre_hook(self.forbidden) for m in
                   (backbone, backbone.student_semantic_link_encoder, backbone.stereo_feature_link,
                    backbone.stereo_feature_link.channel, backbone.build_cost,
                    self.model.map_to_bev_module, self.model.backbone_2d)]
        self.in_reference = True
        try:
            yield
            check(self.calls == before and self.reference_calls == dict(dense_head_2d=1, depth_loss_head=1),
                  'reference call counts or sender/receiver chronology differ')
        finally:
            self.in_reference = False
            for handle in handles: handle.remove()

    def channel(self, module, inputs):
        super().channel(module, inputs)
        check(len(inputs) == 2 and float(inputs[1]) == self.snr, 'public SNR without external receiver arguments')
        self.current_device = inputs[0].device
        self.pending['noise_rng_before'] = rng_record(self.current_device)
        self.pending['channel_input_shape'] = list(inputs[0].shape)

    def channel_done(self, module, inputs, outputs):
        check(self.pending is not None, 'noise audit outside native frame')
        self.pending['noise_rng_after'] = rng_record(self.current_device)
        check((self.pending['noise_rng_before'] == self.pending['noise_rng_after']) == (module.kind == 'identity'), 'actual channel RNG consumption')

    def link_finish(self, module, inputs, outputs):
        super().link_finish(module, inputs, outputs)
        coding = module.last_f9
        check(coding is not None and coding['arm'] == self.arm and not coding['receiver_side_information'], 'sender record/receiver barrier')
        self.pending['F9_coding'] = coding.copy()
        self.pending['arm'] = self.arm


def validate_native_record(row, arm, channel, snr):
    check(row['arm'] == arm and row['channel'] == channel, 'actual native treatment')
    account, coding = row['accounting'], row['F9_coding']
    check(account['channel'] == channel and account['snr_db'] == coding['nominal_snr'] == float(snr), 'actual sender/channel SNR')
    check(account['allocation'] == ('uniform' if arm == 'U' else 'F9_' + arm), 'actual coding arm')
    check(row['channel_input_shape'] == [1, 62400, 2], 'full native symbol layout')
    check(account['data_complex_uses'] == account['total_complex_uses'] == coding['data_complex_uses'] == 62400, 'fixed62400 attempt')
    check(account['stereo_complex_uses'] == 49920 and account['appearance_complex_uses'] == 12480, 'complete stereo/appearance charge')
    check(account['header_complex_uses'] == account['pilot_complex_uses'] == 0, 'identity/AWGN overhead')
    check(account['boundary'] == 'stereo_features_before_receiver_cost', 'native receiver cost boundary')
    check(len(account['tx_energy_per_frame']) == 1 and abs(account['tx_energy_per_frame'][0]/62400-1) <= 1e-4, 'actual full channel energy')
    energy, gains = coding['stereo_group_energy'], coding['amplitude_gains']
    check(len(energy) == len(gains) == 64 and all(math.isfinite(v) and v >= 0 for v in energy), 'complete finite64 group energies')
    check(all(math.isfinite(v) and .5 <= v <= 2 for v in gains), 'bounded actual amplitudes')
    check(math.isclose(sum(energy) + coding['appearance_energy'], coding['actual_energy_float64'], rel_tol=1e-12, abs_tol=1e-8), 'measured group sum')
    check(math.isclose(coding['actual_energy_float64'], account['tx_energy_per_frame'][0], rel_tol=1e-5, abs_tol=1e-3), 'sender/channel measured energy agree')
    check(not coding['receiver_side_information'], 'no gain map reaches receiver')
    if arm == 'U': check(gains == [1.] * 64, 'uniform head bypass')
    if arm in ('P', 'S'):
        check(len(coding['matrix_records']) == 2 and all(m['row_sum_max_error'] <= 6e-5 and m['cross_vertical_mass'] == 0 for m in coding['matrix_records']), 'two-direction correspondence support')
    else: check(coding['matrix_records'] is None, 'generic/uniform has no correspondence')
    for name in ('noise_rng_before', 'noise_rng_after'):
        value = row[name]; check(hashlib.sha256(bytes.fromhex(value['state_hex'])).hexdigest() == value['sha256'], 'complete actual inference RNG state')
    check((row['noise_rng_before'] == row['noise_rng_after']) == (channel == 'identity'), 'actual inference noise convention')
    check(row['sequence'] == ['student', 'student', 'link_start', 'channel', 'link_done', 'build_cost'], 'receive-only chronology')
    check(row['cost_inputs_are_received_features'] and row['appearance_input_is_received_feature'], 'no clean-feature bypass')
    check(not row['autograd_enabled'] and set(row['sensor_input_keys']) <= {'batch_size', 'left_img', 'right_img', 'calib', 'image_shape', 'frame_id'}, 'no gradient/GT/private sender input')
