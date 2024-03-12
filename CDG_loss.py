"""
Loss function for CDG.

Parts of the code adapted from https://github.com/nianticlabs/monodepth2
and https://github.com/valeoai/WoodScape.
Please refer to the license of the above repo.

"""

import numpy as np
import torch
import torch.nn as nn

from utils import bilinear_sampler


class InverseWarp:
    def __init__(self, args):
        self.ego_mask = args.ego_mask
        self.frame_ids = args.frame_ids
        self.num_scales = args.num_scales
        self.min_depth = args.min_depth
        self.max_depth = args.max_depth

    def warp(self, inputs, outputs) -> None:
        raise NotImplementedError("Invalid InverseWarp Attempted!")


    def disp_to_depth(self, disp):
        """Convert network's sigmoid output into depth prediction
        The formula for this conversion is given in the 'additional considerations'
        section of the paper.
        """
        min_disp = 1 / self.max_depth
        max_disp = 1 / self.min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
        return depth



class PhotometricFisheye(InverseWarp):
    def __init__(self, args):
        """Inverse Warp class for Fisheye
        :param args: input params from config file
        """
        super().__init__(args)
        self.crop = args.crop

    @staticmethod
    @torch.jit.script
    def img2world(depth: torch.Tensor, theta_lut, angle_lut, essential_mat) -> torch.Tensor:
        """Transform coordinates in the pixel frame to the camera frame.
        :param depth: depth values for the pixels -- [B x H * W]
        :param theta_lut: Look up table containing coords for angle in the image plane -- [B x H * W x 1]
        :param angle_lut: Look up table containing coords for angle of incidence -- [B x H * W x 1]
        :param essential_mat: The camera transform matrix -- [B x 4 x 4]
        :return: world_coords: The world based coordinates -- [B x 4 x H * W]
        """
        depth = depth.reshape(depth.size(0), 1, -1)  # B x 1 x H * W
        # angle in the image plane
        theta = theta_lut.permute(0, 2, 1)
        # Obtain angle of incidence from radius
        angle_of_incidence = (angle_lut.permute(0, 2, 1)).to(device=depth.device)
        r_world = torch.sin(angle_of_incidence) * depth
        x = r_world * torch.cos(theta)
        y = r_world * torch.sin(theta)
        # Obtain `z` from the norm
        z = torch.cos(angle_of_incidence) * depth
        cam_coords = torch.cat((x, y, z), 1)
        cam_coords = torch.cat(
            [cam_coords, torch.ones(cam_coords.size(0), 1, cam_coords.shape[2]).to(device=depth.device)], 1)
        world_coords = essential_mat @ cam_coords
        return world_coords

    def world2img(self, world_coords: torch.Tensor, intrinsics: torch.Tensor, distortion_coeffs: torch.Tensor,
                  height: int, width: int) -> tuple:
        """Transform 3D world co-ordinates to the pixel frame.
        :param world_coords: The camera based coords -- [B x 4 x H * W]
        :param intrinsics: The camera intrinsics -- [B x 4]
        :param distortion_coeffs: k1, k2, k3, k4 -- [B x 4]
        :param height: image height
        :param width: image width
        :return: pixel_coords, mask: The pixel coordinates corresponding to points -- [B x 2 x H * W], [B x 1 x H * W]
        """
        x_cam, y_cam, z = [world_coords[:, i, :].unsqueeze(1) for i in range(3)]
        # angle in the image plane
        theta = torch.atan2(y_cam, x_cam)
        # radius from angle of incidence
        r = torch.sqrt(x_cam * x_cam + y_cam * y_cam)
        # Calculate angle using z
        a = np.pi / 2 - torch.atan2(z, r)
        distortion_coeffs = distortion_coeffs.unsqueeze(1).unsqueeze(1)
        r_mapping = sum([distortion_coeffs[:, :, :, i] * torch.pow(a, i + 1) for i in range(4)])

        intrinsics = intrinsics.unsqueeze(1).unsqueeze(1)
        x = r_mapping * torch.cos(theta) * intrinsics[:, :, :, 0] + intrinsics[:, :, :, 2]
        y = r_mapping * torch.sin(theta) * intrinsics[:, :, :, 1] + intrinsics[:, :, :, 3]

        x_norm = 2 * x / (width - 1) - 1
        y_norm = 2 * y / (height - 1) - 1
        pcoords_norm = torch.cat([x_norm, y_norm], 1)  # b x 2 x hw
        if self.ego_mask:
            x_mask = (x_norm > -1) & (x_norm < 1)
            y_mask = (y_norm > -1) & (y_norm < 1)
            mask = (x_mask & y_mask).reshape(pcoords_norm.size(0), 1, height, width).float()
        else:
            mask = None
        return pcoords_norm, mask

    def inverse_warp(self, source_img, norm, car_mask,
                     extrinsic_mat, K, D, theta_lut, angle_lut) -> tuple:
        """Inverse warp a source image to the target image plane for fisheye images
        :param source_img: source image (to sample pixels from) -- [B x 3 x H x W]
        :param norm: Depth map of the target image -- [B x 1 x H x W]
        :param extrinsic_mat: DoF pose vector from target to source -- [B x 4 x 4]
        :param K: Camera intrinsic matrix -- [B x 4]
        :param D: Camera distortion co-efficients k1, k2, k3, k4 -- [B x 4]
        :param theta_lut: Look up table containing coords for angle in the image plane -- [B x H * W x 1]
        :param angle_lut: Look up table containing coords for angle of incidence -- [B x H * W x 1]
        :return: Projected source image -- [B x 3 x H x W]
        """
        batch_size, _, height, width = source_img.size()
        norm = norm.reshape(batch_size, height * width)

        world_coords = self.img2world(norm, theta_lut, angle_lut, extrinsic_mat)  # [B x 4 x H * W]
        image_coords, mask = self.world2img(world_coords, K, D, height, width)

        padding_mode = "border" if self.crop else "zeros"
        projected_img = bilinear_sampler(source_img, image_coords, mode='bilinear', padding_mode=padding_mode)

        if not self.crop:
            # TODO: Check interpolation
            # car_hood = bilinear_sampler(car_mask, image_coords, mode='nearest', padding_mode=padding_mode)
            projected_img = projected_img * car_mask

        return projected_img, mask

    def fisheye_inverse_warp(self, inputs, outputs):
        for scale in range(self.num_scales):
            for frame_id in self.frame_ids[1:]:
                disp = outputs[("disp", scale)]
                # disp = F.interpolate(disp, [disp.size(-2)*2**scale, disp.size(-1)*2**scale], mode="bilinear", align_corners=False)
                depth = self.disp_to_depth(disp)

                if not self.crop:
                    car_mask = inputs["mask", 0]
                    depth = depth * car_mask
                    depth[depth == 0] = 0.1
                else:
                    car_mask = None

                image = inputs[("color", frame_id, 0)]
                intrinsic_mat = inputs[("K", frame_id)]
                distortion_coeffs = inputs[("D", frame_id)]
                theta_lut = inputs[("theta_lut", frame_id)]
                angle_lut = inputs[("angle_lut", frame_id)]
                essential_mat = outputs[("cam_T_cam", 0, frame_id)]

                outputs[("color", frame_id, scale)], outputs[("ego_mask", frame_id, scale)] = \
                    self.inverse_warp(image,
                                      depth,
                                      car_mask,
                                      essential_mat,
                                      intrinsic_mat,
                                      distortion_coeffs,
                                      theta_lut,
                                      angle_lut)

    def warp(self, inputs, outputs):
        self.fisheye_inverse_warp(inputs, outputs)



class PhotometricReconstructionLoss(nn.Module):
    def __init__(self, inverse_warp_object: InverseWarp, args):
        """Loss function for unsupervised monocular depth
        :param args: input params from config file
        """
        super().__init__()
        self.warp = inverse_warp_object

        self.frame_ids = args.frame_ids
        self.num_scales = args.num_scales
        self.crop = args.crop
        self.seed = 1e-7

        self.disable_auto_mask = args.disable_auto_mask
        self.clip_loss = args.clip_loss_weight
        self.ssim_weight = args.ssim_weight
        self.reconstr_weight = args.reconstr_weight
        self.smooth_weight = args.disparity_smoothness
        self.self_discover_weight_mask = args.self_discover_weight_mask

    def disp_smoothness(self, disp: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """Smoothens the output depth map
        :param norm: Depth map of the target image -- [B x 1 x H x W]
        :param img: Images from the image_stack -- [B x 3 x H x W]
        :return Mean value of the smoothened image
        """
        disp_gradients_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        disp_gradients_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        image_gradients_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        image_gradients_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        disp_gradients_x *= torch.exp(-image_gradients_x)
        disp_gradients_y *= torch.exp(-image_gradients_y)

        return disp_gradients_x.mean() + disp_gradients_y.mean()

    @staticmethod
    def ssim(x, y):
        """Computes a differentiable structured image similarity measure."""
        x = nn.ReflectionPad2d(1)(x)
        y = nn.ReflectionPad2d(1)(y)
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        mu_x = nn.AvgPool2d(kernel_size=3, stride=1)(x)
        mu_y = nn.AvgPool2d(kernel_size=3, stride=1)(y)
        sigma_x = nn.AvgPool2d(kernel_size=3, stride=1)(x ** 2) - mu_x ** 2
        sigma_y = nn.AvgPool2d(kernel_size=3, stride=1)(y ** 2) - mu_y ** 2
        sigma_xy = nn.AvgPool2d(kernel_size=3, stride=1)(x * y) - mu_x * mu_y
        ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        ssim_d = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
        return torch.clamp((1 - ssim_n / ssim_d) / 2, 0, 1)

    def compute_reprojection_loss(self, predicted, target, ego_mask=None):
        """Computes reprojection loss between predicted and target images"""
        if type(ego_mask) == torch.Tensor:
            l1_loss = (torch.abs(target - predicted) * ego_mask).mean(1, True)
            ssim_error = self.ssim(predicted, target)
            ssim_loss = (ssim_error * ego_mask).mean(1, True)
        else:
            l1_loss = torch.abs(target - predicted).mean(1, True)
            ssim_loss = self.ssim(predicted, target).mean(1, True)

        reprojection_loss = self.ssim_weight * ssim_loss + self.reconstr_weight * l1_loss

        if self.clip_loss:
            mean, std = reprojection_loss.mean(), reprojection_loss.std()
            reprojection_loss = torch.clamp(reprojection_loss, max=float(mean + self.clip_loss * std))

        return reprojection_loss, l1_loss, ssim_loss

    @staticmethod
    def compute_loss_masks(reprojection_loss, identity_reprojection_loss):
        """ Compute loss masks for each of standard reprojection and depth hint
        reprojection"""

        if identity_reprojection_loss is None:
            # we are not using automasking - standard reprojection loss applied to all pixels
            reprojection_loss_mask = torch.ones_like(reprojection_loss)

        else:
            # we are using automasking
            all_losses = torch.cat([reprojection_loss, identity_reprojection_loss], dim=1)
            idxs = torch.argmin(all_losses, dim=1, keepdim=True)
            reprojection_loss_mask = (idxs == 0).float()

        return reprojection_loss_mask

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses"""
        losses = dict()
        total_loss = 0
        target = inputs[("color", 0, 0)]
        for scale in range(self.num_scales):
            loss = 0

            # --- PHOTO-METRIC LOSS ---
            reprojection_loss = list()
            for frame_id in self.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                if self.crop:
                    ego_mask = outputs[("ego_mask", frame_id, scale)]
                else:
                    ego_mask = outputs[("ego_mask", frame_id, scale)] * inputs["mask", 0]
                    outputs[("ego_mask", frame_id, scale)] = ego_mask

                reproj_loss = self.compute_reprojection_loss(pred, target, ego_mask)[0]
                reprojection_loss.append(reproj_loss)
            reprojection_loss = torch.cat(reprojection_loss, 1)

            # --- AUTO MASK ---
            if not self.disable_auto_mask:
                identity_reprojection_loss = list()
                for frame_id in self.frame_ids[1:]:
                    target = inputs[("color", 0, 0)]
                    pred = inputs[("color", frame_id, 0)]
                    reproj_loss = self.compute_reprojection_loss(pred, target)[0]
                    identity_reprojection_loss.append(reproj_loss)
                identity_reprojection_loss = torch.cat(identity_reprojection_loss, 1)
                identity_reprojection_loss, _ = torch.min(identity_reprojection_loss, dim=1, keepdim=True)
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).to(
                    device=identity_reprojection_loss.device) * 1e-5
                reprojection_loss, _ = torch.min(reprojection_loss, dim=1, keepdim=True)

                reprojection_loss_mask = self.compute_loss_masks(reprojection_loss, identity_reprojection_loss)

                # combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                reprojection_loss, _ = torch.min(reprojection_loss, dim=1, keepdim=True)
                reprojection_loss_mask = torch.ones_like(reprojection_loss)

                # combined = reprojection_loss

            # --- COMPUTING MIN FOR MONOCULAR APPROACH ---
            reprojection_loss = reprojection_loss * reprojection_loss_mask
            reprojection_loss = reprojection_loss.sum() / (reprojection_loss_mask.sum() + 1e-7)

            loss += reprojection_loss
            # if combined.shape[1] == 1:
            #     forward_optimise = combined
            # else:
            #     forward_optimise, forward_idxs = torch.min(combined, dim=1)

            # loss += forward_optimise.mean() / (2 ** scale)


            # --- SMOOTHNESS LOSS ---
            disp = outputs[("disp", scale)]
            # disp = F.interpolate(disp, [disp.size(-2) * 2 ** scale, disp.size(-1) * 2 ** scale], mode="bilinear", align_corners=False)
            normalized_disp = (disp / (disp.mean([2, 3], True) + self.seed))
            smooth_loss = self.disp_smoothness(normalized_disp, inputs[("color", 0, 0)])
            loss += self.smooth_weight * smooth_loss / (2 ** scale)

            total_loss += loss
            losses[f"depth_loss/{scale}"] = loss

        total_loss /= self.num_scales
        losses["depth_loss"] = total_loss
        return losses

    def forward(self, inputs, outputs):
        """Loss function for self-supervised norm and pose on monocular videos"""
        self.warp.warp(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)
        return losses, outputs
