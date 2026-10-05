"""Passive F5 evaluation boundary and same-input distortion observation."""
import torch
from .inference import SENSOR_INPUT_KEYS


class StereoFeatureObserver:
    def __init__(self,model,channel,record,compare):
        backbone=model.backbone_3d;link=backbone.stereo_feature_link
        if backbone.semantic_link is not None or link is None or link.channel.kind!=channel:
            raise ValueError('locked F5 stereo feature link required')
        self.record,self.compare=record,compare;self.pending=None;self.received=None
        self.calls=dict(frames=0,student=0,codec=0,channel=0,build_cost=0,forbidden=0)
        self.handles=[]
        for module in (backbone.feature_backbone,backbone.feature_neck,model.lidar_model):
            self.handles.append(module.register_forward_pre_hook(self.forbidden))
        self.handles.extend([backbone.register_forward_pre_hook(self.start),
            backbone.student_semantic_link_encoder.register_forward_hook(self.student),
            link.register_forward_pre_hook(self.link_start),link.channel.register_forward_pre_hook(self.channel),
            link.register_forward_hook(self.link_finish),backbone.build_cost.register_forward_pre_hook(self.cost),
            backbone.register_forward_hook(self.finish)])

    def forbidden(self,module,inputs):
        self.calls['forbidden']+=1;raise RuntimeError('teacher executed during F5 eval')

    def start(self,module,inputs):
        batch=inputs[0]
        if module.training or torch.is_grad_enabled() or self.pending is not None or set(batch)-set(SENSOR_INPUT_KEYS):
            raise RuntimeError('training/autograd/non-sensor/overlapping frame entered F5 eval')
        if batch['batch_size']!=1 or len(batch['frame_id'])!=1: raise RuntimeError('native batch1 required')
        self.pending=dict(frame_id=str(batch['frame_id'][0]),sequence=[],sensor_input_keys=sorted(batch),autograd_enabled=False)

    def event(self,name):
        if self.pending is None: raise RuntimeError('operation outside sensor-only frame')
        self.pending['sequence'].append(name)

    def student(self,module,inputs,outputs):
        if module.training: raise RuntimeError('student training during eval')
        self.event('student');self.calls['student']+=1

    def link_start(self,module,inputs):
        if self.pending is None or self.pending['sequence']!=['student','student'] or module.training:
            raise RuntimeError('link entered before exact two student views')
        self.event('link_start');self.calls['codec']+=1

    def channel(self,module,inputs):
        if self.pending is None or self.pending['sequence']!=['student','student','link_start']:
            raise RuntimeError('channel entered outside link boundary')
        self.event('channel');self.calls['channel']+=1

    def link_finish(self,module,inputs,outputs):
        if self.pending is None or self.pending['sequence']!=['student','student','link_start','channel']:
            raise RuntimeError('decoder output without counted channel')
        self.event('link_done');self.received=outputs
        self.pending['channel']=module.channel.kind
        self.pending['accounting']={k:(v.detach().cpu().tolist() if isinstance(v,torch.Tensor) else v)
                                    for k,v in module.last_accounting.items()}
        with torch.no_grad():
            for branch,reference,received in zip(('left_stereo','right_stereo','appearance'),inputs[:3],outputs):
                self.pending[branch]=self.compare(reference,received)

    def cost(self,module,inputs):
        if self.pending is None or self.pending['sequence']!=['student','student','link_start','channel','link_done']:
            raise RuntimeError('receiver cost construction before communication completion')
        if len(inputs)!=5 or inputs[0] is not self.received[0] or inputs[1] is not self.received[1] or inputs[2:4]!=(None,None):
            raise RuntimeError('receiver cost used a clean/private feature bypass')
        self.event('build_cost');self.calls['build_cost']+=1
        self.pending['cost_inputs_are_received_features']=True

    def finish(self,module,inputs,outputs):
        if self.pending is None or self.pending['sequence']!=['student','student','link_start','channel','link_done','build_cost']:
            raise RuntimeError('incomplete F5 native boundary sequence')
        if outputs['rpn_feature'] is not self.received[2]: raise RuntimeError('receiver appearance feature bypass')
        self.pending['appearance_input_is_received_feature']=True
        self.calls['frames']+=1;self.record(self.pending.copy());self.pending=None;self.received=None

    def close(self):
        for handle in self.handles: handle.remove()
        self.received=None;self.pending=None
