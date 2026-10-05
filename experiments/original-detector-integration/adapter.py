"""Isolated received-only original RGB transport and official Stereo-RCNN preprocessing.

No final checkpoint is loaded or claimed by these engineering helpers. The
formal launcher must enforce the final-stage/source/checkpoint protocol first.
The original model and radio APIs are passed explicitly to avoid collision of
Cao's model.py with Stereo-RCNN's model package in a detector process.
"""
import math
import copy
import numpy as np
import torch


def receiver_rgb(model, observation):
    """Only observation and fixed model parameters cross this receiver entry."""
    received = model.receive_payload(observation)
    shape = tuple(received['original_shape'])
    outputs = model.semantic.decode(received)
    if len(outputs) != 2 or any(value.shape != (1,3,*shape) or value.dtype != torch.float32
                               or not torch.isfinite(value).all() for value in outputs):
        raise RuntimeError('received native RGB layout/dtype/finite differs')
    return outputs, shape


def transmit_rgb(model, radio, left, right, boxes, channel, snr_db=10., generator=None):
    """One air attempt, explicit sender/channel/receiver; erasure retains resources."""
    if torch.is_grad_enabled() or any(module.training for module in model.modules()):
        raise RuntimeError('original RGB inference requires no_grad and allmodules eval')
    if (left.shape != right.shape or left.ndim != 4 or left.shape[:2] != (1,3)
            or left.dtype != torch.float32 or right.dtype != left.dtype or left.device != right.device
            or any(not torch.isfinite(value).all() or value.min()<0 or value.max()>1 for value in (left,right))):
        raise ValueError('finite native paired sensor floatRGB0..1 required')
    if not math.isfinite(snr_db): raise ValueError('finite SNR required')
    masks = radio.roi_masks(left.shape[-2:], boxes, left)
    native_masks = tuple(mask[..., :left.shape[-2], :left.shape[-1]] for mask in masks)
    clean = model.semantic.encode(left,right,native_masks)
    air, account = model.transmit_payload(clean,boxes,channel)
    # Accounting is sender-side reporting only; receiver_rgb never receives it.
    if (account['total_uses'] != len(air) or account['data_uses']+account['control_uses']+account['pilot_uses'] != len(air)
            or account['pilot_uses'] != radio.pilot_count(channel)
            or not math.isclose(account['total_energy'],len(air),rel_tol=1e-4,abs_tol=1e-4)):
        raise RuntimeError('original physical-air accounting differs')
    observation = radio.propagate(air,channel,snr_db,generator)
    del clean,masks,native_masks,air
    try:
        outputs,shape = receiver_rgb(model,observation)
    except radio.FrameErasure as error:
        return dict(outputs=None,accounting=account,erasure=str(error),decoder_range=None)
    if shape != tuple(left.shape[-2:]):
        raise RuntimeError('received sensor geometry differs from attempted frame')
    ranges=[dict(minimum=float(value.min()),maximum=float(value.max()),
                 out_of_range_fraction=float(((value<0)|(value>1)).float().mean())) for value in outputs]
    return dict(outputs=outputs,accounting=account,erasure=None,decoder_range=ranges)


def stereo_rcnn_preprocess(decoded,prep_im_for_blob,pixel_means,target_size,max_size):
    """Official native preprocessing of received floatRGB; no clipping or uint8."""
    if len(decoded)!=2 or decoded[0].shape != decoded[1].shape:
        raise ValueError('equal received stereo pair required')
    if any(value.ndim!=4 or value.shape[:2]!=(1,3) or value.dtype!=torch.float32
           or not torch.isfinite(value).all() for value in decoded):
        raise ValueError('finite batch1 decodedfloatRGB required')
    bgr=[np.ascontiguousarray(value.detach().cpu().numpy()[0].transpose(1,2,0)[...,::-1]*255.,dtype=np.float32)
         for value in decoded]
    shape=tuple(bgr[0].shape)
    left,right,scale=prep_im_for_blob(*bgr,pixel_means,target_size,max_size)
    images=tuple(torch.from_numpy(np.ascontiguousarray(image)).permute(2,0,1).unsqueeze(0).contiguous()
                 for image in (left,right))
    info=torch.tensor([[*left.shape[:2],scale]],dtype=torch.float32)
    return images,info,shape


def liga_preprocess(decoded,calibration,frame_id,augmentor,collate_batch,device='cpu'):
    """Original crop/collation of received RGB with fresh public calibration.

    Keep calib_ori/original image_shape for the native KITTI writer; callers
    must pass only SENSOR_INPUT_KEYS to model inference. No GT/points needed.
    """
    if len(decoded)!=2 or decoded[0].shape!=decoded[1].shape:
        raise ValueError('equal received stereo pair required')
    if any(value.ndim!=4 or value.shape[:2]!=(1,3) or value.dtype!=torch.float32
           or not torch.isfinite(value).all() for value in decoded):
        raise ValueError('finite batch1 decodedfloatRGB required')
    queue=augmentor.data_augmentor_queue
    if len(queue)!=1 or queue[0].func.__name__!='random_crop':
        raise ValueError('native crop-only LIGA evaluation required')
    config=queue[0].keywords['config']
    if tuple(config[k] for k in ('MIN_REL_X','MAX_REL_X','MIN_REL_Y','MAX_REL_Y','MAX_CROP_H','MAX_CROP_W'))!=(0,0,1.,1.,320,1280):
        raise ValueError('locked native evalcrop geometry differs')
    images=[np.ascontiguousarray(value.detach().cpu().numpy()[0].transpose(1,2,0)*255.,dtype=np.float32)
            for value in decoded]
    original_shape=np.asarray(images[0].shape[:2],dtype=np.int32)
    sample=dict(left_img=images[0],right_img=images[1],calib=copy.deepcopy(calibration),
                calib_ori=copy.deepcopy(calibration),frame_id=str(frame_id),image_shape=original_shape.copy())
    sample=augmentor.forward(sample)
    if set(sample)!={'left_img','right_img','calib','calib_ori','frame_id','image_shape'}:
        raise RuntimeError('non-sensor data introduced by native evalcrop')
    # Native Kitti __getitem__ restores raw image_shape AFTER prepare_data.
    sample['image_shape']=original_shape
    batch=collate_batch([sample])
    if set(batch)!={'left_img','right_img','calib','calib_ori','frame_id','image_shape','batch_size'} or batch['batch_size']!=1:
        raise RuntimeError('native sensor-only collation differs')
    for key in ('left_img','right_img'):
        batch[key]=torch.as_tensor(batch[key],dtype=torch.float32,device=device)
        if not torch.isfinite(batch[key]).all():raise RuntimeError('nonfinite received-image preprocessing')
    return batch
