"""
Dataloader for SynWoodScape.

Part of the code is adapted from https://github.com/valeoai/WoodScape.
Please refer to the license of the above repo.

"""

import json
import math
import os
import pickle
import random
import re
from collections import namedtuple

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image  # using pillow-simd for increased speed
from torchvision import transforms


class SynWoodScapeRawDataset(data.Dataset):
    """Fisheye Woodscape Raw dataloader"""

    def __init__(self, data_path=None, path_file=None, is_train=False, config=None):
        super().__init__()

        self.data_path = data_path
        self.image_paths = [line.rstrip('\n') for line in open(path_file)]
        self.is_train = is_train
        self.args = config
        self.task = 'depth'
        self.batch_size = config.batch_size
        self.crop = config.crop
        self.semantic_classes = config.semantic_num_classes
        self.num_scales = config.num_scales
        self.frame_idxs = config.frame_ids
        self.original_res = namedtuple('original_res', 'width height')(1280, 966)
        self.network_input_width = config.width
        self.network_input_height = config.height
        self.total_car1_images = 6054
        self.color_aug = None

        # self.cropped_coords = dict(Car1=dict(FV=(0, 0, 1279, 713),
        #                                      MVL=(266, 0, 1279, 508),
        #                                      MVR=(0, 0, 1008, 479),
        #                                      RV=(0, 0, 1279, 645)),
        #                            Car2=dict(FV=(160, 272, 1030, 677),
        #                                      MVL=(327, 7, 1096, 410),
        #                                      MVR=(175, 4, 935, 404),
        #                                      RV=(285, 187, 1000, 572)))

        self.cropped_coords = dict(Car1=dict(FV=(114, 110, 1176, 610),
                                             MVL=(343, 5, 1088, 411),
                                             MVR=(185, 5, 915, 425),
                                             RV=(186, 203, 1105, 630)),
                                   Car2=dict(FV=(160, 272, 1030, 677),
                                             MVL=(327, 7, 1096, 410),
                                             MVR=(175, 4, 935, 404),
                                             RV=(285, 187, 1000, 572)))

        self.to_tensor = transforms.ToTensor()
        # self.resize = transforms.Resize((self.network_input_height, self.network_input_width),
        #                                 interpolation=transforms.InterpolationMode.BICUBIC)
        self.resize_label = transforms.Resize((self.network_input_height, self.network_input_width),
                                              interpolation=transforms.InterpolationMode.NEAREST)
        
        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.network_input_height // s, self.network_input_width // s),
                                               interpolation=transforms.InterpolationMode.BICUBIC)

        if "depth" in self.task:
            with open('data/LUTs_syn.pkl', 'rb') as f:
                self.LUTs = pickle.load(f)
            with open('data/LUTs_syn_1_4.pkl', 'rb') as f:
                self.LUTs_2 = pickle.load(f)
                

    def scale_intrinsic(self, intrinsics, cropped_coords) -> tuple:
        """Scales the intrinsics from original res to the network's initial input res"""
        D = np.array(intrinsics[4:8], dtype=np.float32)
        K = np.array(intrinsics[:3], dtype=np.float32)
        K = np.insert(K, 0, 1.0)
        K[2] += self.original_res.width / 2
        K[3] += self.original_res.height / 2
        if self.crop:
            # Adjust the offset of the cropped intrinsic around the width and height.
            K[2] -= cropped_coords[0]
            K[3] -= cropped_coords[1]
            # Compensate for resizing
            K[2] *= self.network_input_width / (cropped_coords[2] - cropped_coords[0])
            K[3] *= self.network_input_height / (cropped_coords[3] - cropped_coords[1])
            D *= self.network_input_width / (cropped_coords[2] - cropped_coords[0])
        else:
            D *= self.network_input_width / self.original_res.width
            K[2] *= self.network_input_width / self.original_res.width
            K[3] *= self.network_input_height / self.original_res.height
        return K, D

    def get_displacements_from_speed(self, frame_index, cam_side):
        """get displacement magnitudes using speed and time."""

        with open(os.path.join(self.data_path, "vehicle_data", "previous_images",
                                                        f'{frame_index}.txt'), 'r') as file:
            previous_oxt_file = file.read()

        with open(os.path.join(self.data_path, "vehicle_data", "rgb_images",
                                                        f'{frame_index}.txt'), 'r') as file:
            present_oxt_file = file.read()

        pattern = r'velocity Vector3D\(x=([-+]?\d*\.\d+), y=([-+]?\d*\.\d+), z=([-+]?\d*\.\d+)\)'
        match_previous = re.search(pattern, previous_oxt_file)
        match_present = re.search(pattern, present_oxt_file)

        if match_previous and match_present:
            x_previous = float(match_previous.group(1))
            y_previous = float(match_previous.group(2))
            z_previous = float(match_previous.group(3))

            speed_previous = math.sqrt(x_previous**2 + y_previous**2 + z_previous**2)

            x_present = float(match_present.group(1))
            y_present = float(match_present.group(2))
            z_present = float(match_present.group(3))

            speed_present = math.sqrt(x_present**2 + y_present**2 + z_present**2)

        # Convert km/hr to m/s

        displacement = np.array(0.5 * (speed_previous + speed_present) * 0.1).astype(
            np.float32)

        return displacement

    def get_image(self, index, cropped_coords, frame_index, cam_side):
        recording_folder = "rgb_images" if index == 0 else "previous_images"
        file = f"{frame_index}_{cam_side}.png" if index == 0 else f"{frame_index}_{cam_side}_prev.png"
        path = os.path.join(self.data_path, recording_folder, file)
        image = Image.open(path).convert('RGB')
        if self.crop:
            return image.crop(cropped_coords)
        return image

    def get_label(self, gt_folder, cropped_coords, frame_index, cam_side):
        path = os.path.join(self.data_path, gt_folder, "gtLabels", f"{frame_index}_{cam_side}.png")
        image = Image.open(path).convert('L')
        if self.crop:
            return image.crop(cropped_coords)
        return image

    def get_depth_gts(self, cropped_coords, frame_index, cam_side):
        path = os.path.join(self.data_path, "depth_maps", "raw_data", f"{frame_index}_{cam_side}.npy")
        depth_map = np.load(path)
        image = Image.fromarray(depth_map).convert('L')
        if self.crop:
            return image.crop(cropped_coords)
        return image

    def get_intrinsics(self, cropped_coords, frame_index, cam_side):
        data = json.load(open(os.path.join(self.data_path, "calibration_data",
                                           f"{cam_side}.json")))
        intrinsics = list(data['intrinsic'].values())
        K, D = self.scale_intrinsic(intrinsics, cropped_coords)
        return K, D, intrinsics

    def get_detection_label(self, crop_coords, frame_index, cam_side):
        path = os.path.join(self.data_path, "box_2d_annotations", f"{frame_index}_{cam_side}.txt")
        if os.stat(path).st_size != 0:
            boxes = torch.from_numpy(np.loadtxt(path, delimiter=",", usecols=(1, 2, 3, 4, 5)).reshape(-1, 5))

            # Re-parameterize box for annotation
            w = torch.abs(boxes[:, 3] - boxes[:, 1])
            h = torch.abs(boxes[:, 4] - boxes[:, 2])
            x_c = torch.minimum(boxes[:, 1], boxes[:, 3]) + (w / 2)
            y_c = torch.minimum(boxes[:, 2], boxes[:, 4]) + (h / 2)

            # # Normalize
            w /= self.original_res.width
            x_c /= self.original_res.width
            h /= self.original_res.height
            y_c /= self.original_res.height

            boxes[:, 1] = x_c
            boxes[:, 2] = y_c
            boxes[:, 3] = w
            boxes[:, 4] = h

            if self.crop:
                cropping = dict(left=crop_coords[0], top=crop_coords[1], right=crop_coords[2], bottom=crop_coords[3])
                cropped_boxes = None
                boxes = cropped_boxes
        else:
            boxes = torch.from_numpy(np.zeros((1, 5)))

        targets = torch.zeros((len(boxes), 6))
        targets[:, 1:] = boxes
        return targets

    def to_tensor_depth_gts(self, label: np.array) -> torch.Tensor:
        return torch.Tensor(label)

    def to_tensor_semantic_label(self, label: np.array) -> torch.LongTensor:
        label[label > self.semantic_classes - 1] = 0
        return torch.LongTensor(label)

    @staticmethod
    def to_tensor_motion_label(label: np.array) -> torch.LongTensor:
        label[label > 0] = 1  # Any class greater than 0 is set to 1
        return torch.LongTensor(label)

    def preprocess(self, inputs):
        """Resize color images to the required scales and augment if required.
        Create the color_aug object in advance and apply the same augmentation to all images in this item.
        This ensures that all images input to the pose network receive the same augmentation.
        """
        labels_list = ["depth_gts", "motion_labels", "semantic_labels"]

        for k in list(inputs):
            if "color" in k:
                name, frame_id, _ = k
                for i in range(self.num_scales):
                    inputs[(name, frame_id, i)] = self.resize[i](inputs[(name, frame_id, i - 1)])
            elif any(x in k for x in labels_list):
                name, frame_id, _ = k
                inputs[(name, frame_id, 0)] = self.resize_label(inputs[(name, frame_id, -1)])
            else:
                name, frame_id = k
                inputs[(name, frame_id)] = inputs[(name, frame_id)]

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                name, frame_id, scale = k
                inputs[(name, frame_id, scale)] = self.to_tensor(f)
                inputs[(name + "_aug", frame_id, scale)] = self.to_tensor(self.color_aug(f))
            elif any(x in k for x in labels_list):
                name, frame_id, scale = k
                if name == "depth_gts":
                    inputs[(name, frame_id, scale)] = self.to_tensor_depth_gts(np.array(f))
                elif name == "semantic_labels":
                    inputs[(name, frame_id, scale)] = self.to_tensor_semantic_label(np.array(f))
                elif name == "motion_labels":
                    inputs[(name, frame_id, scale)] = self.to_tensor_motion_label(np.array(f))
            else:
                name, frame_id = k
                if name == "detection_labels":
                    inputs[(name, frame_id)] = f
                else:
                    inputs[(name, frame_id)] = torch.from_numpy(f)

    def destruct_original_image_tensors(self, inputs):
        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]
        if "depth" in self.task:
            del inputs[("depth_gts", 0, -1)]
        if "semantic" in self.task:
            del inputs[("semantic_labels", 0, -1)]
        if "motion" in self.task:
            del inputs[("motion_labels", 0, -1)]

    def create_and_process_training_items(self, index):
        inputs = dict()
        do_color_aug = self.is_train and random.random() > 0.5
        frame_index, cam_side = self.image_paths[index].split('.')[0].split('_')

        if self.crop:
            if int(frame_index[1:]) < self.total_car1_images:
                cropped_coords = self.cropped_coords["Car1"][cam_side]
            else:
                cropped_coords = self.cropped_coords["Car2"][cam_side]
        else:
            cropped_coords = None

        for i in self.frame_idxs:
            inputs[("color", i, -1)] = self.get_image(i, cropped_coords, frame_index, cam_side)

            if "depth" in self.task:
                # if self.is_train:
                if i != 0:
                    inputs[("K", i)], inputs[("D", i)], intrinsics = self.get_intrinsics(cropped_coords,
                                                                                            frame_index, cam_side)
                    k1 = intrinsics[4]
                    inputs[("theta_lut", i)] = self.LUTs[k1]["theta"]
                    inputs[("angle_lut", i)] = self.LUTs[k1]["angle_maps"]
                
        K_2 = inputs[("K", -1)].copy()
        K_2[2] /= 4
        K_2[3] /= 4
        inputs[("K", 2)] = K_2
        inputs[("D", 2)] = inputs[("D", -1)] / 4
        inputs[("theta_lut", 2)] = self.LUTs_2[k1]["theta"]
        inputs[("angle_lut", 2)] = self.LUTs_2[k1]["angle_maps"]

        # inputs[("angle_lut_x", 2)] = self.LUTs_2_x[k1]["angle_maps"]
        # inputs[("angle_lut_y", 2)] = self.LUTs_2_y[k1]["angle_maps"]

        # radius
        x = np.linspace(0, int(self.network_input_width/4) - 1, int(self.network_input_width/4))
        y = np.linspace(0, int(self.network_input_height/4) - 1, int(self.network_input_height/4))
        mesh_x, mesh_y = np.meshgrid(x, y)
        mesh_x, mesh_y = mesh_x.reshape(-1, 1), mesh_y.reshape(-1, 1)

        x_cam = (mesh_x - K_2[2]) / K_2[0]
        y_cam = (mesh_y - K_2[3]) / K_2[0]

        r = np.sqrt(x_cam * x_cam + y_cam * y_cam)

        inputs[("r", 2)] = r

        distortion_map = (inputs[('r', 2)].reshape(1, int(self.network_input_height / 4), int(self.network_input_width / 4)) / \
                          inputs[('angle_lut', 2)].reshape(1, int(self.network_input_height / 4), int(self.network_input_width / 4)))

        inputs[("distortion_map", 2)] = (distortion_map - distortion_map.min() + 1e-8) / (distortion_map.max() - distortion_map.min() + 1e-8)


        # distortion, -1

        x = np.linspace(0, int(self.network_input_width) - 1, int(self.network_input_width))
        y = np.linspace(0, int(self.network_input_height) - 1, int(self.network_input_height))
        mesh_x, mesh_y = np.meshgrid(x, y)
        mesh_x, mesh_y = mesh_x.reshape(-1, 1), mesh_y.reshape(-1, 1)

        x_cam = (mesh_x - inputs[("K", -1)][2]) / inputs[("K", -1)][0]
        y_cam = (mesh_y - inputs[("K", -1)][3]) / inputs[("K", -1)][0]

        r = np.sqrt(x_cam * x_cam + y_cam * y_cam)

        inputs[("r", -1)] = r

        distortion_map = (
                inputs[('r', -1)].reshape(1, int(self.network_input_height), int(self.network_input_width)) / \
                inputs[('angle_lut', -1)].reshape(1, int(self.network_input_height), int(self.network_input_width)))

        inputs[("distortion_map", -1)] = (distortion_map - distortion_map.min() + 1e-8) / (
                    distortion_map.max() - distortion_map.min() + 1e-8)

        if "depth" in self.task:
            inputs[("displacement_magnitude", -1)] = self.get_displacements_from_speed(frame_index, cam_side)
            inputs[("depth_gts", 0, -1)] = self.get_depth_gts(cropped_coords, frame_index, cam_side)

        if "semantic" in self.task:
            inputs[("semantic_labels", 0, -1)] = self.get_label("semantic_annotations", cropped_coords, frame_index,
                                                                cam_side)

        if "motion" in self.task:
            inputs[("motion_labels", 0, -1)] = self.get_label("motion_annotations", cropped_coords, frame_index,
                                                              cam_side)

        if "detection" in self.task:
            inputs[("detection_labels", 0)] = self.get_detection_label(cropped_coords, frame_index, cam_side)

        if do_color_aug:
            self.color_aug = transforms.ColorJitter(brightness=(0.8, 1.2),
                                                    contrast=(0.8, 1.2),
                                                    saturation=(0.8, 1.2),
                                                    hue=(-0.1, 0.1))
        else:
            self.color_aug = (lambda x: x)

        self.preprocess(inputs)
        self.destruct_original_image_tensors(inputs)

        return inputs

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.
        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:
            ("color",          <frame_id>, <scale>)       raw color images,
            ("K",              <frame_id>)                camera intrinsics,
            ("D",              <frame_id>)                distortion coefficients,
            ("angle_lut",      <frame_id>)                look up table containing coords for angle of incidence,
            ("theta_lut",      <frame_id>)                look up table containing coords for angle in the image plane,
            ("color_aug",      <frame_id>)                augmented color image list similar to above raw color list,
            ("displacement_magnitude", -1)                displacement from t-1 to t (reference frame)
            ("displacement_magnitude",  1)                displacement from t+1 to t (reference frame)
            ("motion_labels",  <frame_id>, <scale>        motion segmentation labels of t (reference frame)
            ("semantic_labels",<frame_id>, <scale>)       semantic segmentation labels of t (reference frame)
            ("detection_labels", <frame_id>, <scale>)     detection labels of t (reference frame)

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',

        <scale> is an integer representing the scale of the image relative to the full size image:
           -1       images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        return self.create_and_process_training_items(index)

    def collate_fn(self, batch):
        """Handling the detection_label as each image has a different number of objects so when batch_size > 1,
        the pytorch loader couldn't handle it. So here we stack the bounding boxes to be (#of_object, 6).
        If there is no orientation and to be (#of_object, 7) if the orientation parameters is on.
        :param batch: output returned from __getitem__ function
        :return: return modified version from the batch after edit "detection_label"
        """
        for key in list(batch[0].keys()):
            temp = []
            for i in range(self.batch_size):
                if key == ("detection_labels", 0):
                    batch[i][key][:, 0] = i
                    temp.append(batch[i][key])
                else:
                    temp.append(batch[i][key].unsqueeze(0))
            batch[0][key] = torch.cat(temp, 0)
        return batch[0]
