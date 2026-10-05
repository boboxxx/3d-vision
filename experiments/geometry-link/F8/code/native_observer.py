"""F8 exact parameter scope and native target barrier, with joint student autograd."""
from geocomm.stereo_task_adaptation import NativeTaskAdaptation
from geocomm.stereo_feature_warmup import PREFIX,CONVOLUTIONS
from geocomm.student import StereoStudentEncoder
from torch import nn

def student_names():
    base={'context.0.weight','context.1.weight','context.1.bias','context.3.weight','context.4.weight','context.4.bias',
          'local.0.weight','local.1.weight','local.1.bias','stereo_context.weight','stereo_context.bias',
          'stereo.0.weight','stereo.1.weight','stereo.2.weight','appearance.0.weight','appearance.1.weight','appearance.1.bias'}
    for i in (6,7,8):
        base.update(f'context.{i}.body.{j}.{k}' for j,ks in ((0,('weight',)),(1,('weight','bias')),(3,('weight',)),(4,('weight','bias'))) for k in ks)
    assert len(base)==35
    return {'backbone_3d.student_semantic_link_encoder.'+n for n in base}

def freeze_scope(model,scope):
    expected={PREFIX+branch+'.'+str(index)+'.'+kind for branch,indices in CONVOLUTIONS.items() for index in indices for kind in ('weight','bias')}
    parameters=dict(model.named_parameters());b=model.backbone_3d;link=b.stereo_feature_link;student=b.student_semantic_link_encoder
    if scope not in ('codec','joint') or b.semantic_link is not None or link is None or link.channel.kind!='awgn':raise ValueError('explicit F8 uniform AWGN scope required')
    if type(student) is not StereoStudentEncoder or {'backbone_3d.student_semantic_link_encoder.'+n for n,_ in student.named_parameters()}!=student_names():raise ValueError('exact existing35 student parameters required')
    if {PREFIX+n for n,_ in link.named_parameters()}!=expected or len(expected)!=16:raise ValueError('exact existing16 codec parameters required')
    if scope=='joint':expected|=student_names()
    if not expected<=set(parameters):raise ValueError('missing parameter')
    for module in list(student.modules())+list(link.modules()):
        if isinstance(module,(nn.modules.batchnorm._BatchNorm,nn.modules.dropout._DropoutNd)):raise ValueError('train-dependent layer in eval-only sender/codec')
    model.eval()
    for name,p in parameters.items():p.requires_grad_(name in expected)
    return [(n,p) for n,p in parameters.items() if n in expected]

class ScopedNativeTaskAdaptation(NativeTaskAdaptation):
    def __init__(self,model,scope):
        self.scope=scope
        self.model=model;self.pending=None;self.received=None;self.handles=[];self.gradient_handles=[]
        self.calls=dict(steps=0,student=0,codec=0,channel=0,build_cost=0,map_to_bev=0,BEV=0,head3D=0,forbidden=0)
        b=model.backbone_3d;link=b.stereo_feature_link
        if scope not in ('codec','joint') or b.semantic_link is not None or link is None or link.channel.kind!='awgn':
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

    def student(self,module,inputs,outputs):
        if module.training or any(x.requires_grad!=(self.scope=='joint') for x in outputs):raise RuntimeError('student autograd scope differs')
        self.event('student');self.calls['student']+=1
