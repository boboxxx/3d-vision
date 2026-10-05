"""Explicit F7 channel scope with unchanged shared native3D objective/barrier.

Only constructor hook installation differs from the F6 identity-only observer.
Root F6 code stays untouched while detector engineering is pending.
"""
from geocomm.stereo_task_adaptation import NativeTaskAdaptation
from geocomm.stereo_feature_warmup import PREFIX,CONVOLUTIONS
from torch import nn

def freeze_matched_codec(model,arm):
    expected={PREFIX+branch+'.'+str(index)+'.'+kind for branch,indices in CONVOLUTIONS.items()
              for index in indices for kind in ('weight','bias')}
    parameters=dict(model.named_parameters());link=model.backbone_3d.stereo_feature_link
    if (arm not in ('identity','awgn') or model.backbone_3d.semantic_link is not None
            or link is None or link.channel.kind!=arm or len(expected)!=16
            or {PREFIX+n for n,_ in link.named_parameters()}!=expected or not expected<=set(parameters)):
        raise ValueError('explicit F7 channel and exact16 existing codec parameters required')
    for module in link.modules():
        if isinstance(module,(nn.modules.batchnorm._BatchNorm,nn.modules.dropout._DropoutNd)):
            raise ValueError('locked eval codec cannot contain train-dependent layers')
    model.eval()
    for name,parameter in parameters.items():parameter.requires_grad_(name in expected)
    return [(name,parameter) for name,parameter in parameters.items() if name in expected]

class MatchedNativeTaskAdaptation(NativeTaskAdaptation):
    def __init__(self,model,arm):
        self.model=model;self.pending=None;self.received=None;self.handles=[];self.gradient_handles=[]
        self.calls=dict(steps=0,student=0,codec=0,channel=0,build_cost=0,map_to_bev=0,BEV=0,head3D=0,forbidden=0)
        b=model.backbone_3d;link=b.stereo_feature_link
        if arm not in ('identity','awgn') or b.semantic_link is not None or link is None or link.channel.kind!=arm:
            raise ValueError('explicit fixed F7 native stereo identity/AWGN arm required')
        for module in (b.feature_backbone,b.feature_neck,model.lidar_model,model.dense_head_2d,model.depth_loss_head):
            if module is not None:self.handles.append(module.register_forward_pre_hook(self.forbidden))
        self.handles.extend([b.register_forward_pre_hook(self.backbone_start),
            b.student_semantic_link_encoder.register_forward_hook(self.student),
            link.register_forward_pre_hook(self.link_start),link.channel.register_forward_pre_hook(self.channel),
            link.register_forward_hook(self.link_finish),b.build_cost.register_forward_pre_hook(self.cost_start),
            b.build_cost.register_forward_hook(self.cost_finish),b.register_forward_hook(self.backbone_finish),
            model.map_to_bev_module.register_forward_pre_hook(lambda m,i:self.receiver('map_to_bev',m,i)),
            model.backbone_2d.register_forward_pre_hook(lambda m,i:self.receiver('BEV',m,i)),
            model.dense_head.register_forward_pre_hook(self.head)])
