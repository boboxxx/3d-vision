"""F6 clean native3D objective, eval-only receiver and an explicit target barrier."""
import math
import torch
from .stereo_feature_warmup import freeze_except_codec,assert_frozen
from .inference import SENSOR_INPUT_KEYS

SENSOR_TRAIN_KEYS=set(SENSOR_INPUT_KEYS)|{'random_T'}
SEQUENCE=['student','student','link_start','channel','link_done','build_cost',
          'backbone_done','map_to_bev','BEV','GT_at_3D_head','head3D']


class NativeTaskAdaptation:
    def __init__(self,model):
        self.model=model;self.pending=None;self.received=None;self.handles=[];self.gradient_handles=[]
        self.calls=dict(steps=0,student=0,codec=0,channel=0,build_cost=0,map_to_bev=0,BEV=0,head3D=0,forbidden=0)
        b=model.backbone_3d;link=b.stereo_feature_link
        if b.semantic_link is not None or link is None or link.channel.kind!='identity':
            raise ValueError('locked native stereo identity link required')
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

    def forbidden(self,module,inputs):
        self.calls['forbidden']+=1;raise RuntimeError('teacher/depthloss/2Dhead executed during F6')

    def event(self,event):
        if self.pending is None:raise RuntimeError('operation outside native task step')
        sequence=self.pending['sequence']
        if len(sequence)>=len(SEQUENCE) or event!=SEQUENCE[len(sequence)]:
            raise RuntimeError('native task sequence differs: '+event)
        sequence.append(event)

    def gradient(self,name,value):
        if not value.requires_grad:raise RuntimeError('native communication/cost graph lacks gradient: '+name)
        self.pending['gradient_shapes'][name]=list(value.shape)
        def observe(gradient):
            if self.pending is None or not torch.isfinite(gradient).all():raise RuntimeError('nonfinite/unscoped receiver gradient')
            if name in self.pending['gradient_norms']:raise RuntimeError('duplicate native backward hook: '+name)
            norm=float(torch.linalg.vector_norm(gradient.detach()))
            if not math.isfinite(norm):raise RuntimeError('nonfinite native gradient norm')
            self.pending['gradient_norms'][name]=norm
        self.gradient_handles.append(value.register_hook(observe))

    def backbone_start(self,module,inputs):
        if module.training or set(inputs[0])-SENSOR_TRAIN_KEYS or 'gt_boxes' in inputs[0]:
            raise RuntimeError('GT/training/unauthorized input reached native sender')
        if self.pending is None or self.pending['sequence']:raise RuntimeError('overlapping native backbone step')

    def student(self,module,inputs,outputs):
        if module.training or any(x.requires_grad for x in outputs):raise RuntimeError('student trainable or in training mode')
        self.event('student');self.calls['student']+=1

    def link_start(self,module,inputs):
        if set(inputs[3])-SENSOR_TRAIN_KEYS or module.training:raise RuntimeError('non-sensor data reached link')
        self.event('link_start');self.calls['codec']+=1
        self.pending['feature_shapes']={name:list(value.shape) for name,value in zip(('left_stereo','right_stereo','appearance'),inputs[:3])}

    def channel(self,module,inputs):
        self.event('channel');self.calls['channel']+=1;self.gradient('symbols',inputs[0])

    def link_finish(self,module,inputs,outputs):
        self.event('link_done');self.received=outputs
        for name,value in zip(('left_stereo','right_stereo','appearance'),outputs):self.gradient(name,value)
        account=module.last_accounting
        for key in ('channel','allocation','boundary','data_complex_uses','total_complex_uses','pilot_complex_uses',
                    'header_complex_uses','stereo_complex_uses','appearance_complex_uses','snr_db'):
            self.pending[key]=account[key]
        self.pending.update(tx_energy=float(account['tx_energy_per_frame'][0]),cbr=account['cbr_complex_per_input_real_scalar'])

    def cost_start(self,module,inputs):
        self.event('build_cost');self.calls['build_cost']+=1
        if len(inputs)!=5 or inputs[0] is not self.received[0] or inputs[1] is not self.received[1] or inputs[2:4]!=(None,None):
            raise RuntimeError('clean/private bypass at native stereo cost inputs')
        self.pending['cost_inputs_are_received_features']=True

    def cost_finish(self,module,inputs,outputs):self.gradient('native_cost',outputs)

    def backbone_finish(self,module,inputs,outputs):
        self.event('backbone_done')
        if outputs['rpn_feature'] is not self.received[2] or 'gt_boxes' in outputs:
            raise RuntimeError('appearance bypass or premature target introduction')
        self.pending['appearance_input_is_received_feature']=True

    def receiver(self,event,module,inputs):
        if module.training or 'gt_boxes' in inputs[0]:raise RuntimeError('training/GT before native3D head')
        self.event(event);self.calls['map_to_bev' if event=='map_to_bev' else 'BEV']+=1

    def head(self,module,inputs):
        self.event('head3D');self.calls['head3D']+=1
        if module.training or inputs[0].get('gt_boxes') is not self.gt_boxes:raise RuntimeError('native3D target scope differs')
        self.pending['GT_introduced_only_at_head']=True

    def forward(self,sensors,gt_boxes):
        if self.pending is not None or set(sensors)-SENSOR_TRAIN_KEYS or any(m.training for m in self.model.modules()):
            raise ValueError('native task requires eval-only model and separate sensor/target inputs')
        if not torch.is_grad_enabled() or sensors['batch_size']!=1 or len(sensors['frame_id'])!=1:
            raise ValueError('native task needs batch1 and enabled autograd')
        if gt_boxes.ndim!=3 or gt_boxes.shape[0]!=1 or gt_boxes.shape[2]!=8 or gt_boxes.requires_grad or not torch.isfinite(gt_boxes).all():
            raise ValueError('native augmented3D GT layout/finite differs')
        self.gt_boxes=gt_boxes
        self.pending=dict(sequence=[],gradient_norms={},gradient_shapes={},frame_id=str(sensors['frame_id'][0]),
            sender_input_keys=sorted(sensors),GT_shape=list(gt_boxes.shape),all_modules_eval=True,
            random_T_present='random_T' in sensors,loss_type='native3D_cls_box_direction')
        batch=self.model.backbone_3d(dict(sensors))
        batch=self.model.map_to_bev_module(batch);batch=self.model.backbone_2d(batch)
        if 'gt_boxes' in batch:raise RuntimeError('target introduced before3D head')
        self.event('GT_at_3D_head');batch['gt_boxes']=gt_boxes;self.model.dense_head(batch)
        loss,tb=self.model.dense_head.get_loss()
        if not loss.requires_grad or not torch.isfinite(loss):raise RuntimeError('nonfinite/disconnected native3D loss')
        if set(tb)!={'rpn_loss_cls','rpn_loss_loc','rpn_loss_iou','rpn_loss_dir','rpn_loss'}:
            raise RuntimeError('native3D objective has missing or extra supervision')
        weights=self.model.dense_head.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        self.pending.update(classification_loss=float(tb['rpn_loss_cls']),location_loss=float(tb['rpn_loss_loc']),
            direction_loss=float(tb['rpn_loss_dir']),iou_loss_raw=float(tb['rpn_loss_iou']),
            iou_weight=float(weights['iou_weight']),native_reported_loss=float(tb['rpn_loss']))
        labels=self.model.dense_head.forward_ret_dict['box_cls_labels']
        self.pending.update(empty_GT=gt_boxes.shape[1]==0,positive_anchors=int((labels>0).sum()),
                            background_anchors=int((labels==0).sum()))
        if self.pending['empty_GT'] and (self.pending['positive_anchors']!=0
                or self.pending['background_anchors']<=0
                or any(self.pending[k]!=0 for k in ('location_loss','direction_loss','iou_loss_raw'))):
            raise RuntimeError('native emptyGT background-only target/loss semantics differ')
        expected=self.pending['classification_loss']+self.pending['location_loss']+self.pending['direction_loss']+self.pending['iou_loss_raw']*self.pending['iou_weight']
        if not math.isclose(float(loss.detach()),expected,rel_tol=3e-6,abs_tol=1e-6):raise RuntimeError('native3D component reduction differs')
        return loss

    def finish_backward(self):
        if self.pending is None or self.pending['sequence']!=SEQUENCE or set(self.pending['gradient_norms'])!={'symbols','native_cost','left_stereo','right_stereo','appearance'}:
            raise RuntimeError('incomplete native cost/received-feature/channel backward')
        self.calls['steps']+=1;row=self.pending;self.pending=None;self.received=None;self.gt_boxes=None
        self.model.dense_head.forward_ret_dict.clear()
        for handle in self.gradient_handles:handle.remove()
        self.gradient_handles=[]
        return row

    def close(self):
        for handle in self.handles+self.gradient_handles:handle.remove()
        self.pending=None;self.received=None;self.gt_boxes=None;self.model.dense_head.forward_ret_dict.clear()
