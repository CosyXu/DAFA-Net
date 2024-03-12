"""
Depth encoder model (fisheye cost volume and the Depth Feature Enhancement Module) for FDG.

Parts of the code adapted from https://github.com/nianticlabs/manydepth.
Please refer to the license of the above repo.

"""


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import Img2WorldTransform, World2ImgTransform, SSIM

from models.CDG.resnet_encoder import resnet18
from models.FDG.hr_encoder import hrnet18


class EnhancementEncoder(nn.Module):
    """Encoder adapted to include a cost volume after the 2nd block.

    Setting adaptive_bins=True will recompute the depth bins used for matching upon each
    forward pass - this is required for training from monocular video as there is an unknown scale.
    """

    def __init__(self, pretrained, input_height, input_width,
                 num_depth_bins=32, depth_binning='inverse', cost_volume_mode='l1', device='cpu'):

        super(EnhancementEncoder, self).__init__()

        self.ssim = SSIM()
        self.cost_volume_mode = cost_volume_mode

        self.device = device

        self.depth_binning = depth_binning
        self.set_missing_to_max = True


        self.num_depth_bins = num_depth_bins
        # we build the cost volume at 1/4 resolution
        self.matching_height, self.matching_width = input_height // 4, input_width // 4

        self.is_cuda = False if device == 'cpu' else True
        self.warp_depths = None
        self.warp_depths_dis = None  # to form new cost volume from distortion map


        self.num_ch_enc = np.array([64, 64, 18, 36, 72, 144])
        self.encoder = hrnet18(pretrained=pretrained)
        volume_encoder = resnet18(pretrained=pretrained)
        self.layer0 = nn.Sequential(volume_encoder.conv1, volume_encoder.bn1, volume_encoder.relu)
        self.layer1 = nn.Sequential(volume_encoder.maxpool, volume_encoder.layer1)

        self.backprojector = Img2WorldTransform()
        self.projector = World2ImgTransform()

        if self.is_cuda:
            self.backprojector.to(self.device)
            self.projector.to(self.device)


        self.reduce_conv = nn.Sequential(
            nn.Conv2d(self.num_ch_enc[1] + self.num_depth_bins,
                      out_channels=self.num_ch_enc[1],
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
            )

    def compute_depth_bins(self, prior_depth, scale_facs, is_distortion=False):
        """Compute the depths bins used to build the cost volume. Bins will depend upon
        self.depth_binning, to either be linear in depth (linear) or linear in inverse depth
        (inverse)"""
        with torch.no_grad():
            B, _, H, W = prior_depth.shape
            depth_center = prior_depth  # [1,1,72,136]
            scale_fac = scale_facs.unsqueeze(0).unsqueeze(0)  # [1,1,72,136]
            scheduled_min_depth = depth_center / (1 + scale_fac)
            # scheduled_min_depth = depth_center*(1-scale_fac)
            scheduled_max_depth = depth_center * (1 + scale_fac)

            if self.depth_binning == 'inverse':
                itv = torch.arange(0, self.num_depth_bins, device=prior_depth.device, dtype=prior_depth.dtype,
                                   requires_grad=False).reshape(1, -1, 1, 1) / (self.num_depth_bins - 1)  # 1 D 1 1
                inverse_depth_hypo = 1 / scheduled_max_depth + (
                        1 / scheduled_min_depth - 1 / scheduled_max_depth) * itv  # [1,12,72,136]
                depth_range = 1 / inverse_depth_hypo

            elif self.depth_binning == 'linear':
                itv = torch.arange(0, self.num_depth_bins, device=prior_depth.device, dtype=prior_depth.dtype,
                                   requires_grad=False).reshape(1, -1, 1, 1) / (self.num_depth_bins - 1)  # 1 D H W
                depth_range = scheduled_min_depth + (scheduled_max_depth - scheduled_min_depth) * itv

            elif self.depth_binning == 'log':
                itv = []
                for K in range(self.num_depth_bins):
                    K_ = torch.FloatTensor([K])
                    itv.append(torch.exp(
                        torch.log(torch.FloatTensor([0.1])) + torch.log(torch.FloatTensor([1 / 0.1])) * K_ / (
                                    self.num_depth_bins - 1)))
                itv = torch.FloatTensor(itv).unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(B, 1, H, W).to(
                    scheduled_min_depth.device)  # B D H W
                depth_range = scheduled_min_depth + (scheduled_max_depth - scheduled_min_depth) * itv

            else:
                raise NotImplementedError

        if is_distortion:
            self.warp_depths_dis = depth_range.permute(1, 0, 2, 3)
            if self.is_cuda:
                self.warp_depths_dis = self.warp_depths_dis.to(self.device)
        else:
            self.warp_depths = depth_range.permute(1, 0, 2, 3)
            if self.is_cuda:
                self.warp_depths = self.warp_depths.to(self.device)

    def match_features(self, current_feats, lookup_feats, theta_lut, angle_lut, lookup_poses, intrinsics,
                       distortion_coeffs, is_distortion=False):
        """Compute a cost volume based on L1 difference between current_feats and lookup_feats.

        We backwards warp the lookup_feats into the current frame using the estimated relative
        pose, known intrinsics and using hypothesised depths self.warp_depths (which are either
        linear in depth or linear in inverse depth).

        # current_feats: [2,64,72,136]    lookup_feats: [2,1,64,72,136]

        If relative_pose == 0 then this indicates that the lookup frame is missing (i.e. we are
        at the start of a sequence), and so we skip it"""

        batch_cost_volume = []  # store all cost volumes of the batch
        cost_volume_masks = []  # store locations of '0's in cost volume for confidence

        for batch_idx in range(len(current_feats)):

            volume_shape = (self.num_depth_bins, self.matching_height, self.matching_width)
            cost_volume = torch.zeros(volume_shape, dtype=torch.float, device=current_feats.device)
            counts = torch.zeros(volume_shape, dtype=torch.float, device=current_feats.device)

            # select an item from batch of ref feats
            _lookup_feats = lookup_feats[batch_idx:batch_idx + 1]
            _lookup_poses = lookup_poses[batch_idx:batch_idx + 1]

            if is_distortion:
                _warp_depths = self.warp_depths_dis[:, batch_idx:batch_idx + 1]
            else:
                _warp_depths = self.warp_depths[:, batch_idx:batch_idx + 1]

            _theta_lut = theta_lut[batch_idx:batch_idx + 1]
            _angle_lut = angle_lut[batch_idx:batch_idx + 1]
            _intrinsics = intrinsics[batch_idx:batch_idx + 1]
            _distortion_coeffs = distortion_coeffs[batch_idx:batch_idx + 1]

            world_points = self.backprojector(_warp_depths, _theta_lut,
                                              _angle_lut, _lookup_poses[:, 0])  # [96,4,9792]

            # loop through ref images adding to the current cost volume
            for lookup_idx in range(_lookup_feats.shape[1]):
                lookup_feat = _lookup_feats[:, lookup_idx]  # 1 x C x H x W
                lookup_pose = _lookup_poses[:, lookup_idx]

                # ignore missing images
                if lookup_pose.sum() == 0:
                    continue

                lookup_feat = lookup_feat.repeat([self.num_depth_bins, 1, 1, 1])  # [96,64,72,136]
                pix_locs = self.projector(world_points, _intrinsics, _distortion_coeffs, self.matching_height,
                                          self.matching_width)  # [96,72,136,2]
                warped = F.grid_sample(lookup_feat, pix_locs, padding_mode='zeros', mode='bilinear',
                                       align_corners=True)

                # mask values landing outside the image (and near the border)
                # we want to ignore edge pixels of the lookup images and the current image
                # because of zero padding
                # Masking of ref image border
                x_vals = (pix_locs[..., 0].detach() / 2 + 0.5) * (
                        self.matching_width - 1)  # convert from (-1, 1) to pixel values
                y_vals = (pix_locs[..., 1].detach() / 2 + 0.5) * (self.matching_height - 1)

                edge_mask = (x_vals >= 2.0) * (x_vals <= self.matching_width - 2) * \
                            (y_vals >= 2.0) * (y_vals <= self.matching_height - 2)
                edge_mask = edge_mask.float()

                # masking of current image
                current_mask = torch.zeros_like(edge_mask)
                current_mask[:, 2:-2, 2:-2] = 1.0
                edge_mask = edge_mask * current_mask

                if self.cost_volume_mode == 'l1':
                    diffs = torch.abs(warped - current_feats[batch_idx:batch_idx + 1]).mean(1) * edge_mask
                elif self.cost_volume_mode == 'dot':
                    diffs = warped * current_feats[batch_idx:batch_idx + 1].mean(1) * edge_mask
                elif self.cost_volume_mode == 'ssim':
                    diffs = self.ssim(warped + .5,
                                      current_feats[batch_idx:batch_idx + 1].expand(self.num_depth_bins, -1, -1,
                                                                                    -1) + .5).mean(1) * edge_mask
                elif self.cost_volume_mode == 'ssim_l1':
                    ssim = self.ssim(warped + .5,
                                     current_feats[batch_idx:batch_idx + 1].expand(self.num_depth_bins, -1, -1,
                                                                                   -1) + .5)
                    l1 = torch.abs(warped - current_feats[batch_idx:batch_idx + 1])
                    diffs = (0.85 * (1 - ssim) + 0.15 * l1).mean(1) * edge_mask

                # integrate into cost volume
                counts = counts + (diffs > 0).float()

                cost_volume = cost_volume + diffs

            missing_val_mask = (cost_volume == 0).float()
            # average over lookup images
            cost_volume = cost_volume / (counts + 1e-7)
            if self.cost_volume_mode == 'l1' or self.cost_volume_mode == 'ssim_l1':
                # if some missing values for a pixel location (i.e. some depths landed outside) then
                # set to max of existing values
                if self.set_missing_to_max:
                    cost_volume = cost_volume * (1 - missing_val_mask) + \
                                  cost_volume.max(0)[0].unsqueeze(0) * missing_val_mask
            elif self.cost_volume_mode == 'dot' or self.cost_volume_mode == 'ssim':
                if self.set_missing_to_max:
                    cost_volume = cost_volume * (1 - missing_val_mask) + \
                                  cost_volume.min(0)[0].unsqueeze(0) * missing_val_mask

            batch_cost_volume.append(cost_volume)
            cost_volume_masks.append(missing_val_mask)

        batch_cost_volume = torch.stack(batch_cost_volume, 0)
        cost_volume_masks = torch.stack(cost_volume_masks, 0)

        return batch_cost_volume, cost_volume_masks


    def feature_extraction(self, image, return_all_feats=False):
        """ Run feature extraction on an image - first 2 blocks of ResNet"""

        image = (image - 0.45) / 0.225  # imagenet normalisation
        feats_0 = self.layer0(image)
        feats_1 = self.layer1(feats_0)

        if return_all_feats:
            return [feats_0, feats_1]
        else:
            return feats_1

    def indices_to_disparity(self, indices):
        """Convert cost volume indices to 1/depth for visualisation"""

        depth = torch.gather(self.warp_depths.permute(1, 0, 2, 3), 1, indices.unsqueeze(1))[:, 0].cpu()
        disp = 1 / depth
        return disp

    def compute_confidence_mask(self, cost_volume, num_bins_threshold=None):
        """ Returns a 'confidence' mask based on how many times a depth bin was observed"""

        if num_bins_threshold is None:
            num_bins_threshold = self.num_depth_bins
        confidence_mask = ((cost_volume > 0).sum(1) == num_bins_threshold).float()

        return confidence_mask

    def forward_test(self, current_image, prior_depth, scale_facs):

        # feature extraction
        self.features = self.encoder.shallow_forward(current_image)  # [4,64,144,272] [4,64,72,136]

        # feature extraction on lookup images - disable gradients to save memory
        with torch.no_grad():
            self.compute_depth_bins(prior_depth, scale_facs)

        volume_shape = (current_image.shape[0], self.num_depth_bins, self.matching_height, self.matching_width)
        cost_volume = torch.zeros(volume_shape, dtype=torch.float, device=self.device)

        post_matching_feats = self.reduce_conv(torch.cat([self.features[-1], cost_volume], 1))  # [4,64,72,136]

        self.features = [self.features[-2]] + self.encoder.deep_forward(post_matching_feats)

        return self.features

    def forward(self, current_image, lookup_images,
                theta_lut, angle_lut, essential_mat,
                intrinsics, distortion_coeffs,
                prior_depth, scale_facs):

        # feature extraction

        self.features = self.encoder.shallow_forward(current_image)  # [4,64,144,272] [4,64,72,136]
        current_feats = self.feature_extraction(current_image, return_all_feats=False)  # [4,64,144,272] [4,64,72,136]


        # feature extraction on lookup images - disable gradients to save memory
        with torch.no_grad():
            self.compute_depth_bins(prior_depth, scale_facs)

            batch_size, num_frames, chns, height, width = lookup_images.shape
            lookup_images = lookup_images.reshape(batch_size * num_frames, chns, height, width)  # [4,3,288,544]

            lookup_feats = self.feature_extraction(lookup_images, return_all_feats=False)  # [4,64,72,136]

            _, chns, height, width = lookup_feats.shape
            lookup_feats = lookup_feats.reshape(batch_size, num_frames, chns, height, width)  # [4,1,64,72,136]

            # warp features to find cost volume

            cost_volume, missing_mask = \
                self.match_features(current_feats, lookup_feats, theta_lut, angle_lut, essential_mat, intrinsics,
                                    distortion_coeffs)  # [4,96,72,136] [4,96,72,136]
            depth_m_reg_raw = None
            depth_m_reg = None

            confidence_mask = self.compute_confidence_mask(cost_volume.detach() *
                                                           (1 - missing_mask.detach()))  # [4,72,136]

        # for visualisation - ignore 0s in cost volume for minimum
        if self.cost_volume_mode == 'l1' or self.cost_volume_mode == 'ssim_l1':
            viz_cost_vol = cost_volume.clone().detach()
            viz_cost_vol[viz_cost_vol == 0] = 100
            mins, argmin = torch.min(viz_cost_vol, 1)
            lowest_cost = self.indices_to_disparity(argmin)  # [4,72,136]

        elif self.cost_volume_mode == 'dot' or self.cost_volume_mode == 'ssim':
            viz_cost_vol = cost_volume.clone().detach()
            maxs, argmax = torch.max(viz_cost_vol, 1)
            lowest_cost = self.indices_to_disparity(argmax)  # [4,72,136]

        # mask the cost volume based on the confidence
        cost_volume *= confidence_mask.unsqueeze(1)  # [4,96,72,136]

        post_matching_feats = self.reduce_conv(torch.cat([self.features[-1], cost_volume], 1))  # [4,64,72,136]

        self.features = [self.features[-2]] + self.encoder.deep_forward(post_matching_feats)

        return self.features, lowest_cost, confidence_mask, depth_m_reg_raw, depth_m_reg

