"""Fixed clean-feature pooling interventions, explicitly not communication."""
import hashlib
import torch
import torch.nn.functional as F

CONDITIONS=('control','cost_only','appearance_only','both','depth_preserved')
SENSOR_KEYS={'batch_size','left_img','right_img','calib','image_shape','frame_id'}


def pool_restore(value,strides):
    if value.ndim!=len(strides)+2 or any(s<1 or int(s)!=s for s in strides):
        raise ValueError('pooling rank/strides differ')
    shape=tuple((size+stride-1)//stride for size,stride in zip(value.shape[2:],strides))
    if len(strides)==3: pooled=F.adaptive_avg_pool3d(value,shape);mode='trilinear'
    elif len(strides)==2: pooled=F.adaptive_avg_pool2d(value,shape);mode='bilinear'
    else: raise ValueError('unsupported pooling rank')
    return F.interpolate(pooled,size=value.shape[2:],mode=mode,align_corners=False),list(pooled.shape)


def state_hashes(model):
    result={}
    for name,value in model.state_dict().items():
        cpu=value.detach().cpu().contiguous()
        result[name]=hashlib.sha256(str(cpu.dtype).encode()+str(tuple(cpu.shape)).encode()+cpu.numpy().tobytes()).hexdigest()
    return result


class PoolingDiagnostic:
    def __init__(self,model,condition,record):
        if condition not in CONDITIONS: raise ValueError('unknown locked condition')
        backbone=model.backbone_3d
        if backbone.semantic_link is not None or backbone.student_semantic_link_encoder is None:
            raise ValueError('requires explicit uncompressed frozen student')
        self.condition=condition;self.record=record;self.pending=None
        self.calls=dict(frames=0,student=0,raw_cost=0,forbidden=0)
        self.handles=[]
        for module in [backbone.feature_backbone,backbone.feature_neck,model.lidar_model]:
            self.handles.append(module.register_forward_pre_hook(self.forbidden))
        self.handles.extend([backbone.register_forward_pre_hook(self.start),
            backbone.student_semantic_link_encoder.register_forward_hook(self.student),
            backbone.dres0.register_forward_pre_hook(self.cost),backbone.register_forward_hook(self.finish)])

    def forbidden(self,module,inputs):
        self.calls['forbidden']+=1;raise RuntimeError('original teacher executed during pooling diagnostic')

    def start(self,module,inputs):
        batch=inputs[0]
        if module.training or torch.is_grad_enabled() or self.pending is not None or set(batch)-SENSOR_KEYS:
            raise RuntimeError('training/autograd/non-sensor/overlapping frame entered pooling diagnostic')
        if batch['batch_size']!=1 or len(batch['frame_id'])!=1:
            raise RuntimeError('diagnostic requires native batch1')
        self.pending=dict(frame_id=str(batch['frame_id'][0]),condition=self.condition,student_calls=0,raw_cost_calls=0,
            sensor_input_keys=sorted(batch),appearance_operation='identity',cost_operation='identity',
            communication_enabled=False,autograd_enabled=False)

    def student(self,module,inputs,outputs):
        if self.pending is None or module.training: raise RuntimeError('student outside eval frame')
        self.calls['student']+=1;self.pending['student_calls']+=1
        if self.pending['student_calls']>2: raise RuntimeError('unexpected additional student call')
        if self.pending['student_calls']==2: return None
        stereo,appearance=outputs;self.pending['appearance_shape']=list(appearance.shape)
        self.pending['appearance_pooled_shape']=None
        if self.condition in ('appearance_only','both'):
            restored,shape=pool_restore(appearance,(4,4))
            self.pending.update(appearance_operation='adaptive_avg_pool_4_4_bilinear_restore',appearance_pooled_shape=shape)
            return stereo,restored
        return None

    def cost(self,module,inputs):
        if self.pending is None or module.training or self.pending['student_calls']!=2:
            raise RuntimeError('cost outside locked native eval sequence')
        self.calls['raw_cost']+=1;self.pending['raw_cost_calls']+=1
        if self.pending['raw_cost_calls']!=1 or len(inputs)!=1: raise RuntimeError('unexpected native cost calls')
        value=inputs[0];self.pending['cost_shape']=list(value.shape);self.pending['cost_pooled_shape']=None
        if self.condition in ('cost_only','both','depth_preserved'):
            strides=(1,4,4) if self.condition=='depth_preserved' else (8,4,4)
            restored,shape=pool_restore(value,strides)
            self.pending.update(cost_operation='adaptive_avg_pool_'+'_'.join(map(str,strides))+'_trilinear_restore',cost_pooled_shape=shape)
            return (restored,)
        return None

    def finish(self,module,inputs,outputs):
        if self.pending is None or (self.pending['student_calls'],self.pending['raw_cost_calls'])!=(2,1):
            raise RuntimeError('incomplete per-frame diagnostic sequence')
        self.calls['frames']+=1;self.record(self.pending.copy());self.pending=None

    def close(self):
        for handle in self.handles: handle.remove()
