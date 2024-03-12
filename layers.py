"""
Modules and layers for DAFA-Net.

Parts of the code adapted from https://github.com/valeoai/WoodScape.
Please refer to the license of the above repo.

"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth



class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class Img2WorldTransform(nn.Module):
    def __init__(self):
        super(Img2WorldTransform, self).__init__()

    def forward(self, depth, theta_lut, angle_lut, essential_mat):
        batch_size = depth.size(0)
        depth = depth.reshape(batch_size, 1, -1)  # B x 1 x H * W
        # angle in the image plane
        theta = theta_lut.permute(0, 2, 1)
        # Obtain angle of incidence from radius
        angle_of_incidence = angle_lut.permute(0, 2, 1)
        r_world = torch.sin(angle_of_incidence) * depth
        x = r_world * torch.cos(theta)
        y = r_world * torch.sin(theta)
        # Obtain `z`
        z = torch.cos(angle_of_incidence) * depth
        cam_coords = torch.cat((x, y, z), 1)
        cam_coords = torch.cat(
            [cam_coords, torch.ones(cam_coords.size(0), 1, cam_coords.shape[2]).to(device=cam_coords.device)], 1)
        world_coords = essential_mat @ cam_coords

        return world_coords

class World2ImgTransform(nn.Module):
    def __init__(self):
        super(World2ImgTransform, self).__init__()

    def forward(self, world_coords, intrinsics, distortion_coeffs, height, width):
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
        pcoords_norm = pcoords_norm.permute(0, 2, 1).reshape(world_coords.size(0), int(height), int(width), 2)

        return pcoords_norm


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()




class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


# attention layers

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // ratio, in_planes, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, in_feature):
        x = in_feature
        b, c, _, _ = in_feature.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        out = avg_out
        return self.sigmoid(out).expand_as(in_feature) * in_feature


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, in_feature):
        x = in_feature
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        # x = avg_out
        # x = max_out
        x = self.conv1(x)
        return self.sigmoid(x).expand_as(in_feature) * in_feature

class EnhancementAttentionModule(nn.Module):
    def __init__(self, high_feature_channel, low_feature_channels, output_channel=None, relu=True):
        super(EnhancementAttentionModule, self).__init__()
        in_channel = high_feature_channel + low_feature_channels
        out_channel = high_feature_channel
        if output_channel is not None:
            out_channel = output_channel
        channel = in_channel
        # self.cbam = CBAM(channel)
        self.ca = ChannelAttention(channel)
        self.sa = SpatialAttention()
        self.conv_se = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1)
        if relu:
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Tanh()

    def forward(self, high_features, low_features):
        features = [upsample(high_features)]
        features += low_features
        features = torch.cat(features, 1)

        features = self.ca(features)
        features = self.sa(features)

        return self.act(self.conv_se(features))


# pose utils

def pose_vec2mat(axisangle, translation, invert=False, rotation_mode='euler'):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix"""
    if rotation_mode == "euler":
        R = euler2mat(axisangle)
    elif rotation_mode == "quat":
        R = quat2mat(axisangle)

    t = translation.clone()
    if invert:
        R = R.transpose(1, 2)
        t *= -1
    T = get_translation_matrix(t)
    if invert:
        essential_mat = torch.matmul(R, T)
    else:
        essential_mat = torch.matmul(T, R)
    return essential_mat


@torch.jit.script
def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix"""
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)
    t = translation_vector.contiguous().view(-1, 3, 1)
    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t
    return T


@torch.jit.script
def euler2mat(angle):
    """Convert euler angles to rotation matrix.
    :param angle: rotation angle along 3 axis (in radians) -- [B x 1 x 3]
    :return Rotation matrix corresponding to the euler angles -- [B x 4 x 4]
    """
    batch_size = angle.size(0)
    x, y, z = angle[:, :, 0], angle[:, :, 1], angle[:, :, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z * 0
    ones = zeros + 1
    zmat = torch.stack([cosz, -sinz, zeros, zeros,
                        sinz, cosz, zeros, zeros,
                        zeros, zeros, ones, zeros,
                        zeros, zeros, zeros, ones], dim=1).reshape(batch_size, 4, 4)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros, siny, zeros,
                        zeros, ones, zeros, zeros,
                        -siny, zeros, cosy, zeros,
                        zeros, zeros, zeros, ones], dim=1).reshape(batch_size, 4, 4)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros, zeros,
                        zeros, cosx, -sinx, zeros,
                        zeros, sinx, cosx, zeros,
                        zeros, zeros, zeros, ones], dim=1).reshape(batch_size, 4, 4)

    rotMat = xmat @ ymat @ zmat
    return rotMat


@torch.jit.script
def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    :param quat: quat: first three coeffs of quaternion are rotations.
    fourth is then computed to have a norm of 1 -- size = [B x 1 x 3]
    :return: Rotation matrix corresponding to the quaternion -- size = [B x 4 x 4]
    """
    batch_size = quat.size(0)
    norm_quat = torch.cat([quat[:, :, :1] * 0 + 1, quat], dim=2)
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=2, keepdim=True)
    w, x, y, z = norm_quat[:, :, 0], norm_quat[:, :, 1], norm_quat[:, :, 2], norm_quat[:, :, 3]
    zeros = z * 0
    ones = zeros + 1

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rot_mat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, zeros,
                           2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx, zeros,
                           2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2, zeros,
                           zeros, zeros, zeros, ones], dim=1).reshape(batch_size, 4, 4)
    return rot_mat
