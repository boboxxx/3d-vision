"""F9 scopes preserve the verified native 3D loss and receiver chronology."""
import importlib.util
from pathlib import Path
import time

import torch
from torch import nn

from coupling import ARMS, CorrespondenceLink
from geocomm.stereo_feature_warmup import PREFIX, CONVOLUTIONS
from geocomm.student import StereoStudentEncoder

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location('sealed_F8_observer', ROOT / 'experiments/geometry-link/F8/code/native_observer.py')
sealed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sealed)


def freeze_scope(model, arm):
    b = model.backbone_3d
    link, student = b.stereo_feature_link, b.student_semantic_link_encoder
    if arm not in ARMS or type(link) is not CorrespondenceLink or link.arm != arm or link.channel.kind != 'awgn' or b.semantic_link is not None:
        raise ValueError('exact installed F9 AWGN arm required')
    codec = {PREFIX + branch + '.' + str(index) + '.' + kind
             for branch, indices in CONVOLUTIONS.items() for index in indices for kind in ('weight', 'bias')}
    head = {PREFIX + 'gain_head.' + name for name in ('0.weight', '0.bias', '2.weight', '2.bias')}
    if type(student) is not StereoStudentEncoder or {"backbone_3d.student_semantic_link_encoder." + n for n, _ in student.named_parameters()} != sealed.student_names():
        raise ValueError('exact35 original student parameters required')
    if {PREFIX + n for n, _ in link.named_parameters()} != codec | head or len(codec) != 16:
        raise ValueError('exact16 codec and four head tensors required')
    for module in [*student.modules(), *link.modules()]:
        if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.modules.dropout._DropoutNd)):
            raise ValueError('train-mode dependent sender or codec')
    expected = codec | sealed.student_names() | (head if arm != 'U' else set())
    parameters = dict(model.named_parameters())
    if not expected <= set(parameters) or len(expected) != (51 if arm == 'U' else 55):
        raise ValueError('declared F9 parameter scope')
    model.eval()
    for name, parameter in parameters.items():
        parameter.requires_grad_(name in expected)
    return [(name, parameter) for name, parameter in parameters.items() if name in expected]


class ScopedNativeTaskAdaptation(sealed.ScopedNativeTaskAdaptation):
    def __init__(self, model, arm):
        if arm not in ARMS or type(model.backbone_3d.stereo_feature_link) is not CorrespondenceLink or model.backbone_3d.stereo_feature_link.arm != arm:
            raise ValueError('explicit F9 installed arm required')
        self.arm = arm
        self.sender_context = None
        self.sender_started = None
        self.sender_events = None
        super().__init__(model, 'joint')

    def backbone_start(self, module, inputs):
        super().backbone_start(module, inputs)
        if self.calls['steps'] == 0:
            self.sender_context = torch.profiler.record_function('F9_sender')
            self.sender_context.__enter__()
            self.sender_started = time.perf_counter()
            if inputs[0]['left_img'].is_cuda:
                self.sender_events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                self.sender_events[0].record()

    def channel(self, module, inputs):
        if self.sender_context is not None:
            self.sender_context.__exit__(None, None, None)
            self.sender_context = None
            elapsed_cuda = None
            if self.sender_events is not None:
                self.sender_events[1].record()
                self.sender_events[1].synchronize()
                elapsed_cuda = self.sender_events[0].elapsed_time(self.sender_events[1])
            b = self.model.backbone_3d
            link = b.stereo_feature_link
            self.pending['sender_profile'] = dict(
                scope='first_actual_step_from_sender_backbone_entry_to_channel_input',
                wall_seconds=time.perf_counter() - self.sender_started,
                cuda_elapsed_ms=elapsed_cuda,
                student_parameters=sum(p.numel() for p in b.student_semantic_link_encoder.parameters()),
                source_codec_parameters=sum(p.numel() for branch in (link.stereo_encoder, link.appearance_encoder) for p in branch.parameters()),
                new_head_parameters=561, new_head_executed=self.arm != 'U',
                correspondence_executed=self.arm in ('P', 'S'),
                correspondence_candidate_shifts_per_direction=49 if self.arm in ('P', 'S') else 0,
                limitations='Cold first actual forward with audits/profiling; no warm deployment-latency or complete-FLOPs claim.')
        super().channel(module, inputs)

    def link_finish(self, module, inputs, outputs):
        super().link_finish(module, inputs, outputs)
        if module.last_f9 is None or module.last_f9['arm'] != self.arm or module.last_f9['receiver_side_information']:
            raise RuntimeError('F9 sender record or receiver barrier differs')

    def close(self):
        if self.sender_context is not None:
            self.sender_context.__exit__(None, None, None)
            self.sender_context = None
        super().close()
