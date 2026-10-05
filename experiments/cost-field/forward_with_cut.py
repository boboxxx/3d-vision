"""Isolated author forward with one charged cost/appearance cut and receiver neck."""
from liga.models.backbones_3d_stereo.liga_backbone import *

def forward_with_cut(self, batch_dict):
    left = batch_dict['left_img']
    right = batch_dict['right_img']
    calib = batch_dict['calib']
    fu_mul_baseline = torch.as_tensor(
        [x.fu_mul_baseline for x in calib], dtype=torch.float32, device=left.device)
    if self.boxes_gt_in_cam2_view:
        calibs_Proj = torch.as_tensor(
            [x.K3x4 for x in calib], dtype=torch.float32, device=left.device)
    else:
        calibs_Proj = torch.as_tensor(
            [x.P2 for x in calib], dtype=torch.float32, device=left.device)

    N = batch_dict['batch_size']

    # feature extraction
    if self.student_semantic_link_encoder is None:
        left_features = self.feature_backbone(left)
        left_features = [left] + list(left_features)
        right_features = self.feature_backbone(right)
        right_features = [right] + list(right_features)
        left_stereo_feat, left_sem_feat = self.feature_neck(left_features)
        right_stereo_feat, _ = self.feature_neck(right_features)
    else:
        left_stereo_feat, left_sem_feat = self.student_semantic_link_encoder(left)
        right_stereo_feat, _ = self.student_semantic_link_encoder(right)
        if self.training and self.student_distill_weight > 0:
            from geocomm.student import teacher_feature_targets, feature_distillation
            targets = teacher_feature_targets(self.feature_backbone, self.feature_neck, left, right)
            batch_dict['communication_student_loss'] = self.student_distill_weight * feature_distillation(
                (left_stereo_feat, right_stereo_feat, left_sem_feat), targets)

    if self.stereo_feature_link is not None:
        left_stereo_feat, right_stereo_feat, left_sem_feat = self.stereo_feature_link(
            left_stereo_feat, right_stereo_feat, left_sem_feat, batch_dict)

    # stereo matching: build stereo volume (receiver after feature channel)
    downsampled_depth = self.downsampled_depth.cuda()
    downsampled_disp = fu_mul_baseline[:, None] / \
        downsampled_depth[None, :] / (self.downsample_disp if not self.fullres_stereo_feature else 1)

    cost_raw = self.build_cost(left_stereo_feat, right_stereo_feat,
                               None, None, downsampled_disp)

    # stereo matching network
    if self.semantic_link is not None and self.semantic_link_boundary == 'raw_cost':
        cost_raw, left_sem_feat = self.semantic_link(
            cost_raw, left_sem_feat, batch_dict, downsampled_depth)
    cost0 = self.dres0(cost_raw)
    cost0 = self.dres1(cost0) + cost0
    if self.semantic_link is not None and self.semantic_link_boundary == 'processed_cost':
        cost0, left_sem_feat = self.semantic_link(
            cost0, left_sem_feat, batch_dict, downsampled_depth)

    if len(self.hg_stereo) > 0:
        all_costs = []
        cur_cost = cost0
        for hg_stereo_module in self.hg_stereo:
            cost_residual, _, _ = hg_stereo_module(cur_cost, None, None)
            cur_cost = cur_cost + cost_residual
            all_costs.append(cur_cost)
    else:
        all_costs = [cost0]
    assert len(all_costs) > 0, 'at least one hourglass'

    all_costs[-1], left_sem_feat = self._communication_callback(all_costs[-1], left_sem_feat, batch_dict)
    if self.sem_neck is not None:
        batch_dict['sem_features'] = self.sem_neck([left_sem_feat])
    else:
        batch_dict['sem_features'] = [left_sem_feat]

    batch_dict['rpn_feature'] = left_sem_feat

    # stereo matching: outputs
    batch_dict['depth_preds'] = []
    if not self.training:
        batch_dict['depth_preds_local'] = []
    batch_dict['depth_volumes'] = []
    batch_dict['depth_samples'] = self.depth.clone().detach().cuda()
    for idx in range(len(all_costs)):
        upcost_i, cost_softmax_i, pred_i = self.pred_depth(self.pred_stereo[idx], all_costs[idx], left.shape[2:4])
        batch_dict['depth_volumes'].append(upcost_i)
        batch_dict['depth_preds'].append(pred_i)
        if not self.training:
            batch_dict['depth_preds_local'].append(self.get_local_depth(cost_softmax_i))

    # beginning of 3d detection part
    if self.use_stereo_out_type == "feature":
        out = all_costs[-1]
    elif self.use_stereo_out_type == "prob":
        out = cost_softmax_i.unsqueeze(1)
    elif self.use_stereo_out_type == "cost":
        out = upcost_i.unsqueeze(1)
    else:
        raise ValueError('wrong self.use_stereo_out_type option')
    out_prob = cost_softmax_i

    # convert plane-sweep into 3d volume
    coordinates_3d = self.coordinates_3d.cuda()
    batch_dict['coord'] = coordinates_3d
    norm_coord_imgs = []
    coord_imgs = []
    valids2d = []
    for i in range(N):
        c3d = coordinates_3d.view(-1, 3)
        if 'random_T' in batch_dict:
            random_T = batch_dict['random_T'][i]
            c3d = torch.matmul(c3d, random_T[:3, :3].T) + random_T[:3, 3]
        # in pseudo lidar coord
        c3d = project_pseudo_lidar_to_rectcam(c3d)
        coord_img = project_rect_to_image(
            c3d,
            calibs_Proj[i].float().cuda())

        coord_img = torch.cat(
            [coord_img, c3d[..., 2:]], dim=-1)
        coord_img = coord_img.view(*self.coordinates_3d.shape[:3], 3)

        coord_imgs.append(coord_img)

        img_shape = batch_dict['image_shape'][i]
        valid_mask_2d = (coord_img[..., 0] >= 0) & (coord_img[..., 0] <= img_shape[1]) & \
            (coord_img[..., 1] >= 0) & (coord_img[..., 1] <= img_shape[0])
        valids2d.append(valid_mask_2d)

        # TODO: crop augmentation
        crop_x1, crop_x2 = 0, left.shape[3]
        crop_y1, crop_y2 = 0, left.shape[2]
        norm_coord_img = (coord_img - torch.as_tensor([crop_x1, crop_y1, self.CV_DEPTH_MIN], device=coord_img.device)) / torch.as_tensor(
            [crop_x2 - 1 - crop_x1, crop_y2 - 1 - crop_y1, self.CV_DEPTH_MAX - self.CV_DEPTH_MIN], device=coord_img.device)
        norm_coord_img = norm_coord_img * 2. - 1.
        norm_coord_imgs.append(norm_coord_img)
    norm_coord_imgs = torch.stack(norm_coord_imgs, dim=0)
    coord_imgs = torch.stack(coord_imgs, dim=0)
    valids2d = torch.stack(valids2d, dim=0)

    batch_dict['norm_coord_imgs'] = norm_coord_imgs
    batch_dict['coord_imgs'] = coord_imgs

    valids = valids2d & (norm_coord_imgs[..., 2] >= -1.) & (norm_coord_imgs[..., 2] <= 1.)
    batch_dict['valids'] = valids
    valids = valids.float()

    # Retrieve Voxel Feature from Cost Volume Feature
    Voxel = F.grid_sample(out, norm_coord_imgs, align_corners=True)
    Voxel = Voxel * valids[:, None, :, :, :]

    if (self.voxel_attentionbydisp or
            (self.img_feature_attentionbydisp and self.cat_img_feature)):
        pred_disp = F.grid_sample(out_prob.detach()[:, None],
                                  norm_coord_imgs, align_corners=True)
        pred_disp = pred_disp * valids[:, None, :, :, :]

        if self.voxel_attentionbydisp:
            Voxel = Voxel * pred_disp

    # Retrieve Voxel Feature from 2D Img Feature
    if self.cat_img_feature:
        RPN_feature = left_sem_feat

        norm_coord_imgs_2d = norm_coord_imgs.clone().detach()
        norm_coord_imgs_2d[..., 2] = 0
        Voxel_2D = F.grid_sample(RPN_feature.unsqueeze(2), norm_coord_imgs_2d, align_corners=True)
        Voxel_2D = Voxel_2D * valids2d.float()[:, None, :, :, :]

        if self.img_feature_attentionbydisp:
            Voxel_2D = Voxel_2D * pred_disp

        if Voxel is not None:
            Voxel = torch.cat([Voxel, Voxel_2D], dim=1)
        else:
            Voxel = Voxel_2D

    # (64, 190, 20, 300)
    Voxel = self.rpn3d_convs(Voxel)  # (64, 190, 20, 300)
    batch_dict['volume_features_nopool'] = Voxel

    Voxel = self.rpn3d_pool(Voxel)  # [B, C, Nz, Ny, Nx] in cam view

    batch_dict['volume_features'] = Voxel

    return batch_dict
