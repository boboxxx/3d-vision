"""Headless adaptation of official Stereo-RCNN demo's complete 3D postprocess.

Import only in a process whose sys.path includes the pinned Stereo-RCNN lib.
No LiDAR, label, visualization, or alternative detector participates.
"""
import math
import numpy as np
import torch


def kitti_line(prediction, calibration):
    """Exact released write_detection_results convention, including 1.57."""
    pos = prediction['location']
    dim = prediction['dimensions_w_h_l']
    yaw = prediction['rotation_y']
    alpha = yaw-math.pi/2+math.atan2(-pos[0],pos[2])
    values = [alpha,*prediction['bbox'],dim[1],dim[0],dim[2],
              pos[0]-calibration.t_cam2_cam0[0],pos[1],pos[2],yaw-1.57,prediction['score']]
    return 'Car -1 -1 ' + '%f %f %f %f %f %f %f %f %f %f %f %f %f \n' % tuple(values)


def decode_3d(output, left_image, right_image, info, original_shape, calibration, threshold=.05):
    from model.utils.config import cfg
    from model.roi_layers import nms
    from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes, kpts_transform_inv, border_transform_inv
    from model.utils import kitti_utils, box_estimator
    from model.dense_align import dense_align

    rois_left, rois_right, scores, bbox, dimensions, kpts, left, right = output[:8]
    scale = info[0,2]
    # Six regressands share y/height between left and right; each class is separate.
    bbox = bbox.reshape(1,-1,2,6)
    delta_left = bbox[:,:,:,[0,1,2,3]].reshape(-1,4)
    delta_right = bbox[:,:,:,[4,1,5,3]].reshape(-1,4)
    std = bbox.new_tensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)
    mean = bbox.new_tensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
    delta_left = (delta_left*std+mean).reshape(1,-1,8)
    delta_right = (delta_right*std+mean).reshape(1,-1,8)
    dimensions = dimensions.reshape(-1,5)*bbox.new_tensor(cfg.TRAIN.DIM_NORMALIZE_STDS)+bbox.new_tensor(cfg.TRAIN.DIM_NORMALIZE_MEANS)
    dimensions = dimensions.reshape(-1,10)
    max_prob, delta_kpt = kpts.reshape(-1,4*cfg.KPTS_GRID).max(1)
    delta_left_border = left.reshape(-1,cfg.KPTS_GRID).argmax(1)
    delta_right_border = right.reshape(-1,cfg.KPTS_GRID).argmax(1)
    boxes_left = rois_left[:,:,1:5]
    boxes_right = rois_right[:,:,1:5]
    pred_left = clip_boxes(bbox_transform_inv(boxes_left,delta_left,1),info,1).squeeze(0)/scale
    pred_right = clip_boxes(bbox_transform_inv(boxes_right,delta_right,1),info,1).squeeze(0)/scale
    pred_kpts, types = kpts_transform_inv(boxes_left,delta_kpt.reshape(1,-1,1),cfg.KPTS_GRID)
    border_left = border_transform_inv(boxes_left,delta_left_border.reshape(1,-1,1),cfg.KPTS_GRID)
    border_right = border_transform_inv(boxes_left,delta_right_border.reshape(1,-1,1),cfg.KPTS_GRID)
    keypoints = torch.cat((pred_kpts/scale,types,max_prob.reshape(1,-1,1),border_left/scale,border_right/scale),2).squeeze(0)
    scores = scores.squeeze(0)[:,1]
    indices = (scores>threshold).nonzero().flatten()
    order = scores[indices].argsort(descending=True)
    indices = indices[order]
    boxes_left = pred_left[indices,4:8]
    boxes_right = pred_right[indices,4:8]
    scores = scores[indices]
    dimensions = dimensions[indices,5:10]
    keypoints = keypoints[indices]
    keep = nms(boxes_left,scores,cfg.TEST.NMS).to(device=scores.device,dtype=torch.long)
    boxes_left, boxes_right, scores = boxes_left[keep],boxes_right[keep],scores[keep]
    dimensions, keypoints = dimensions[keep],keypoints[keep]
    if scores.numel() == 0:
        return [], dict(detections_2d=0,initial_solutions=0,dense_solutions=0)
    detections = torch.cat((boxes_left,scores[:,None]),1)
    inferred = torch.as_tensor(kitti_utils.infer_boundary(original_shape,detections.cpu().numpy()),
                               device=scores.device,dtype=scores.dtype)
    for i in range(len(scores)):
        if keypoints[i,4]-keypoints[i,3] < .5*(inferred[i,1]-inferred[i,0]):
            keypoints[i,3:5] = inferred[i]
    boxes_all, kpts_all, poses_all = [],[],[]
    for i in range(len(scores)):
        dim = dimensions[i,:3].cpu().numpy()
        alpha = math.atan2(float(dimensions[i,3]),float(dimensions[i,4]))
        status, state = box_estimator.solve_x_y_z_theta_from_kpt(original_shape,calibration,alpha,
            dim,boxes_left[i].cpu().numpy(),boxes_right[i].cpu().numpy(),keypoints[i].cpu().numpy())
        if status > 0:
            boxes_all.append(detections[i])
            kpts_all.append(keypoints[i])
            poses_all.append(scores.new_tensor([*state[:3],*dim,state[3],alpha]))
    counts = dict(detections_2d=len(scores),initial_solutions=len(poses_all),dense_solutions=0)
    if not poses_all:
        return [],counts
    boxes_all,kpts_all,poses_all = map(torch.stack,(boxes_all,kpts_all,poses_all))
    success, disparity = dense_align.align_parallel(calibration,scale,left_image,right_image,
        boxes_all[:,:4],kpts_all,poses_all[:,:7])
    predictions = []
    for i in range(len(success)):
        if success[i] <= 0:
            continue
        rectified,z = box_estimator.solve_x_y_theta_from_kpt(original_shape,calibration,
            poses_all[i,7].cpu().numpy(),poses_all[i,3:6].cpu().numpy(),boxes_all[i,:4].cpu().numpy(),
            disparity[i].cpu().numpy(),kpts_all[i].cpu().numpy())
        row = dict(score=float(boxes_all[i,4]),bbox=boxes_all[i,:4].cpu().tolist(),
                   dimensions_w_h_l=poses_all[i,3:6].cpu().tolist(),
                   location=[float(rectified[0]),float(rectified[1]),float(z)],
                   rotation_y=float(rectified[2]),alpha=float(poses_all[i,7]),disparity=float(disparity[i]))
        if not np.isfinite([row['score'],*row['bbox'],*row['dimensions_w_h_l'],*row['location'],row['rotation_y'],row['alpha'],row['disparity']]).all():
            raise RuntimeError('nonfinite full Stereo-RCNN 3D solution')
        predictions.append(row)
    counts['dense_solutions'] = len(predictions)
    return predictions,counts
