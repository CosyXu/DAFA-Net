"""
Pose estimation training and depth training utils for DAFA-Net.

Parts of the code adapted from https://github.com/valeoai/WoodScape.
Please refer to the license of the above repo.

"""

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm
from matplotlib.colors import ListedColormap


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



def bilinear_sampler(im: torch.Tensor, flow_field: torch.Tensor,
                     mode='bilinear', padding_mode='border') -> torch.Tensor:
    """Perform bilinear sampling on im given list of x, y coordinates.
    Implements the differentiable sampling mechanism with bilinear kernel in https://arxiv.org/abs/1506.02025.
    flow_field is the tensor specifying normalized coordinates [-1, 1] to be sampled on im.
    For example, (-1, -1) in (x, y) corresponds to pixel location (0, 0) in im,
    and (1, 1) in (x, y) corresponds to the bottom right pixel in im.
    :param im: Batch of images with shape -- [B x 3 x H x W]
    :param flow_field: Tensor of normalized x, y coordinates in [-1, 1], with shape -- [B x 2 x H * W]
    :param mode: interpolation mode to calculate output values 'bilinear' | 'nearest'.
    :param padding_mode: "zeros" use "0" for out-of-bound grid locations,
           padding_mode="border: use border values for out-of-bound grid locations,
           padding_mode="reflection": use values at locations reflected by the border
                        for out-of-bound grid locations.
    :return: Sampled image with shape -- [B x 3 x H x W]
    """
    batch_size, channels, height, width = im.shape
    flow_field = flow_field.permute(0, 2, 1).reshape(batch_size, height, width, 2)
    output = F.grid_sample(im, flow_field, mode=mode, padding_mode=padding_mode, align_corners=True)
    return output


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]"""
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higher resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0, 1, low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0, max_value, resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:, i]) for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)


# Support only preceptually uniform sequential colormaps
# https://matplotlib.org/examples/color/colormaps_reference.html
COLORMAPS = dict(plasma=cm.get_cmap('plasma', 10000),
                 magma=high_res_colormap(cm.get_cmap('magma')),
                 viridis=cm.get_cmap('viridis', 10000))


def tensor2array(tensor, colormap='magma'):
    norm_array = normalize_image(tensor).detach().cpu()
    array = COLORMAPS[colormap](norm_array).astype(np.float32)
    return array.transpose(2, 0, 1)
