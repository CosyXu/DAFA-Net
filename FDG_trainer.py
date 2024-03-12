"""
Depth Estimation training for DAFA-Net.

Parts of the code adapted from https://github.com/valeoai/WoodScape.
Please refer to the license of the above repo.

"""

import random
import time

import numpy as np

import torch
import torch.nn.functional as F
from colorama import Fore, Style
from torch.utils.data import DataLoader

from data_loader.woodscape_loader import WoodScapeRawDataset
from data_loader.synwoodscape_loader import SynWoodScapeRawDataset

from models.CDG.depth_decoder import ResDecoder, SwinDecoder
from models.CDG.pose_decoder import PoseDecoder
from models.CDG.resnet_encoder import ResnetEncoder
from models.CDG.swin_encoder import get_orgwintrans_backbone
from models.FDG.hr_encoder import hrnet18
from models.FDG.enhancement_encoder import EnhancementEncoder
from models.FDG.recurrent_decoder import RecurrentDecoder

from train_utils import TrainUtils

from layers import Img2WorldTransform, World2ImgTransform, disp_to_depth, get_smooth_loss, SSIM, compute_depth_errors, pose_vec2mat


class DepthEstimationModelBase(TrainUtils):
    def __init__(self, args):
        super().__init__(args)

        self.depth_bin_facs = torch.full(size=(self.args.height // 4, self.args.width // 4),
                                         fill_value=self.args.depth_bin_fac).to(self.device)

        # check the frames we need the dataloader to load
        frames_to_load = self.args.frame_ids.copy()
        self.matching_ids = [0]
        if self.args.use_future_frame:
            self.matching_ids.append(1)
        for idx in range(-1, -1 - self.args.num_matching_frames, -1):
            self.matching_ids.append(idx)
            if idx not in frames_to_load:
                frames_to_load.append(idx)

        print('Loading frames: {}'.format(frames_to_load))

        self.models["encoder"] = EnhancementEncoder(self.args.weights_init == "pretrained",
            input_height=self.args.height, input_width=self.args.width,
            depth_binning=self.args.depth_binning, num_depth_bins=self.args.num_depth_bins,
            cost_volume_mode=self.args.cost_volume_mode, device=self.device).to(
            self.device)

        self.models["encoder_context"] = hrnet18(self.args.weights_init == "pretrained").to(self.device)

        self.models["depth"] = RecurrentDecoder(
            num_ch_enc=self.models["encoder"].num_ch_enc,
            iters=self.args.iters).to(self.device)

        self.parameters_to_train += list(self.models["encoder"].parameters())
        self.parameters_to_train += list(self.models["encoder_context"].parameters())
        self.parameters_to_train += list(self.models["depth"].parameters())

        if 'orgSwin' in self.args.encoder_model_type:
            self.models["coarse_encoder"] = get_orgwintrans_backbone(self.args.swin_model_type, pretrained=False).to(self.device)
            self.models["coarse_depth"] = SwinDecoder(self.models["coarse_encoder"].num_ch_enc).to(self.device)

        elif 'Res' in self.args.encoder_model_type:
            self.models["coarse_encoder"] = ResnetEncoder(num_layers=self.args.network_layers, pretrained=False).to(self.device)
            self.models["coarse_depth"] = ResDecoder(self.models["coarse_encoder"].num_ch_enc).to(self.device)
        else:
            raise NotImplementedError

        if self.train_teacher_and_pose:
            self.parameters_to_train += list(self.models["coarse_encoder"].parameters())
            self.parameters_to_train += list(self.models["coarse_depth"].parameters())

        self.models["pose_encoder"] = ResnetEncoder(num_layers=self.args.pose_network_layers,
                                                    pretrained=False,
                                                    num_input_images=2).to(self.device)

        self.models["pose"] = PoseDecoder(self.models["pose_encoder"].num_ch_enc,
                                          num_input_features=1,
                                          num_frames_to_predict_for=2).to(self.device)

        if self.train_teacher_and_pose:
            self.parameters_to_train += list(self.models["pose_encoder"].parameters())
            self.parameters_to_train += list(self.models["pose"].parameters())

        print(f"{Fore.BLUE}=> Training on the {args.dataset.upper()}\n"
              f"=> Models and tensorboard events files are saved to: {args.output_directory} \n")

        # --- Load Data ---
        if 'synwoodscape' in self.args.dataset:
            self.train_dataset = SynWoodScapeRawDataset(data_path=args.dataset_dir,
                                                        path_file=args.train_file,
                                                        is_train=True,
                                                        config=args)

            val_dataset = SynWoodScapeRawDataset(data_path=args.dataset_dir,
                                                 path_file=args.val_file,
                                                 is_train=False,
                                                 config=args)

        elif 'woodscape' in self.args.dataset:
            self.train_dataset = WoodScapeRawDataset(data_path=args.dataset_dir,
                                                     path_file=args.train_file,
                                                     is_train=True,
                                                     config=args)

            val_dataset = WoodScapeRawDataset(data_path=args.dataset_dir,
                                              path_file=args.val_file,
                                              is_train=False,
                                              config=args)

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=args.num_workers,
                                       pin_memory=True,
                                       drop_last=True)

        print(f"{Fore.RED}=> Total number of training examples: {len(self.train_dataset)}{Style.RESET_ALL}")

        self.val_loader = DataLoader(val_dataset,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=args.num_workers,
                                     pin_memory=True,
                                     drop_last=True)

        print(f"{Fore.YELLOW}=> Total number of validation examples: {len(val_dataset)}{Style.RESET_ALL}")

        self.num_total_steps = len(self.train_dataset) // args.batch_size * args.epochs

        if not self.args.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}

        for scale in self.args.scales:
            self.backproject_depth[scale] = Img2WorldTransform().to(self.device)
            self.project_3d[scale] = World2ImgTransform().to(self.device)

        self.depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

    def pre_init(self):
        if self.args.pretrained_weights:
            self.load_model()

        if 'cuda' in self.device:
            torch.cuda.synchronize()

    def freeze_teacher(self):
        if self.train_teacher_and_pose:
            self.train_teacher_and_pose = False
            print('freezing teacher and pose networks!')

            self.parameters_to_train = []
            self.parameters_to_train += list(self.models["encoder"].parameters())
            self.parameters_to_train += list(self.models["encoder_context"].parameters())
            self.parameters_to_train += list(self.models["depth"].parameters())
            self.model_optimizer = torch.optim.NAdam(self.parameters_to_train, self.args.learning_rate)
            self.model_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.model_optimizer, self.args.scheduler_step_size, 0.1)

            self.set_eval()

            self.set_train()

    def depth_train(self):

        for self.epoch in range(self.args.epochs):

            if self.epoch == self.args.freeze_teacher_epoch:
                self.freeze_teacher()

            self.set_train()
            data_loading_time = 0
            gpu_time = 0
            before_op_time = time.time()

            for batch_idx, inputs in enumerate(self.train_loader):
                data_loading_time += (time.time() - before_op_time)
                before_op_time = time.time()
                # -- PUSH INPUTS DICT TO DEVICE --
                self.inputs_to_device(inputs)

                # -- DEPTH ESTIMATION --
                outputs, losses = self.predict_depths(inputs, is_train=True)

                # -- COMPUTE GRADIENT AND DO OPTIMIZER STEP --
                self.optimizer.zero_grad()
                losses["loss"].backward()
                self.optimizer.step()

                duration = time.time() - before_op_time
                gpu_time += duration

                if batch_idx % self.args.log_frequency == 0:
                    self.log_time(batch_idx, duration, losses["loss"].cpu().data,
                                  data_loading_time, gpu_time)

                    self.log("train", inputs, outputs, losses, True)

                    self.val()

                if self.step == self.args.freeze_teacher_step:
                    self.freeze_teacher()

                self.step += 1
                before_op_time = time.time()

            self.lr_scheduler.step()

            if (self.epoch + 1) % self.args.save_frequency == 0 and self.epoch > 15:
                self.save_model()

        # self.save_model()
        print("Training finished!")

    def predict_depths(self, inputs, is_train=False):
        coarse_outputs = dict()
        outputs = dict()

        # predict poses for all frames
        if self.train_teacher_and_pose:
            pose_pred = self.predict_poses(inputs, None)
        else:
            with torch.no_grad():
                pose_pred = self.predict_poses(inputs, None)
        outputs.update(pose_pred)
        coarse_outputs.update(pose_pred)

        # grab poses + frames and stack for input to the FDG
        relative_poses = [inputs[('relative_pose', idx)] for idx in self.matching_ids[1:]]
        relative_poses = torch.stack(relative_poses, 1)

        lookup_frames = [inputs[('color_aug', idx, 0)] for idx in self.matching_ids[1:]]
        lookup_frames = torch.stack(lookup_frames, 1)  # batch x frames x 3 x h x w

        # apply static frame and zero cost volume augmentation
        batch_size = len(lookup_frames)
        augmentation_mask = torch.zeros([batch_size, 1, 1, 1]).to(self.device).float()
        if is_train and not self.args.no_matching_augmentation:
            for batch_idx in range(batch_size):
                rand_num = random.random()
                # static camera augmentation -> overwrite lookup frames with current frame
                if rand_num < 0.25:
                    replace_frames = \
                        [inputs[('color', 0, 0)][batch_idx] for _ in self.matching_ids[1:]]
                    replace_frames = torch.stack(replace_frames, 0)
                    lookup_frames[batch_idx] = replace_frames
                    augmentation_mask[batch_idx] += 1
                # missing cost volume augmentation -> set all poses to 0, the cost volume will skip these frames
                elif rand_num < 0.5:
                    relative_poses[batch_idx] *= 0
                    augmentation_mask[batch_idx] += 1
        outputs['augmentation_mask'] = augmentation_mask

        # CDG path
        if self.train_teacher_and_pose:
            feats = self.models["coarse_encoder"](inputs[("color_aug", 0, 0)])
            coarse_outputs.update(self.models['coarse_depth'](feats))
        else:
            with torch.no_grad():
                feats = self.models["coarse_encoder"](inputs[("color_aug", 0, 0)])
                coarse_outputs.update(self.models['coarse_depth'](feats))

        self.generate_images_pred_for_coarse_model(inputs, coarse_outputs)
        coarse_losses = self.compute_losses_for_coarse_model(inputs, coarse_outputs)

        # update FDG outputs dictionary with CDG outputs
        for key in list(coarse_outputs.keys()):
            _key = list(key)
            if _key[0] in ['depth', 'disp']:
                _key[0] = 'coarse_' + key[0]
                _key = tuple(_key)
                outputs[_key] = coarse_outputs[key]


        # FDG path
        encoder_output, lowest_cost, confidence_mask, _, _ = self.models["encoder"](inputs[("color_aug", 0, 0)],
                                                                                  lookup_frames,
                                                                                  inputs[("theta_lut", 2)],
                                                                                  inputs[("angle_lut", 2)],
                                                                                  relative_poses,
                                                                                  inputs[("K", 2)],
                                                                                  inputs[("D", 2)],
                                                                                  prior_depth=F.interpolate(outputs[('coarse_depth', 0, 0)], [self.args.height//4, self.args.width//4], mode="bilinear").clone().detach(),
                                                                                  scale_facs=self.depth_bin_facs.clone().detach())


        context_output = self.models["encoder_context"](inputs[("color_aug", 0, 0)])
        outputs.update(self.models["depth"](encoder_output, context_output))

        outputs["lowest_cost"] = F.interpolate(lowest_cost.unsqueeze(1),
                                               [self.args.height, self.args.width],
                                               mode="nearest")[:, 0]
        outputs["consistency_mask"] = F.interpolate(confidence_mask.unsqueeze(1),
                                                    [self.args.height, self.args.width],
                                                    mode="nearest")[:, 0]

        if not self.args.disable_motion_masking:
            outputs["consistency_mask"] = (outputs["consistency_mask"] *
                                           self.compute_matching_mask(outputs))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        # update losses with coarse-level losses
        if self.train_teacher_and_pose:
            for key, val in coarse_losses.items():
                try:
                    losses[key] += val
                except:
                    pass

        # update adaptive depth bins
        self.update_adaptive_depth_bins(inputs, outputs)

        return outputs, losses


    def update_adaptive_depth_bins(self, inputs, outputs):
        """Update the current estimates of min/max depth using exponental weighted average"""
        delta_d, _ = torch.max(inputs[("distortion_map", 2)][:, 0], 0)

        delta_d = delta_d * self.args.delta_fac_d

        depth_prior = F.interpolate(outputs[('coarse_depth', 0, 0)].detach(),
                                (self.args.height // 4, self.args.width // 4), mode="bilinear", align_corners=True)[:, 0]

        depth_m = F.interpolate(outputs[('depth', 0, self.args.iters - 1)].detach(),
                                (self.args.height // 4, self.args.width // 4), mode="bilinear", align_corners=True)[: ,0]

        delta, _ = torch.max(torch.cat((depth_m / depth_prior, depth_prior / depth_m), 0), 0)
        # print(delta.shape)
        delta = (delta - 1) * self.args.delta_fac
        # delta = torch.clip(delta, min=self.opt.delta_min, max=self.opt.delta_max)
        self.depth_bin_facs = self.depth_bin_facs * 0.99 + ((1 + delta_d.float()) * delta) * 0.01




    def predict_poses(self, inputs, features=None):
        """Predict poses between input frames."""
        outputs = dict()
        if self.num_pose_frames == 2:

            pose_feats = {f_i: inputs[("color_aug", f_i, 0)] for f_i in self.args.frame_ids}

            for f_i in self.args.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]

                    axisangle, translation = self.models["pose"](pose_inputs)    #[2,2,1,3]  [2,2,1,3]

                    # Normalize the translation vec and multiply by the displacement magnitude obtained from speed
                    # of the vehicle to scale it to the real world translation
                    if self.args.use_velocity:

                        translation_magnitude = translation[:, 0].squeeze(1).norm(p="fro",
                                                                                  dim=1).unsqueeze(1).unsqueeze(2)
                        translation_norm = translation[:, 0] / translation_magnitude
                        translation_norm *= inputs[("displacement_magnitude", f_i)].unsqueeze(1).unsqueeze(2)
                        translation = translation_norm

                    else:
                        translation = translation[:, 0]

                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = pose_vec2mat(axisangle[:, 0],
                                                                       translation,
                                                                       invert=(f_i < 0),
                                                                       rotation_mode=self.args.rotation_mode)

            # now we need poses for matching - compute without gradients
            pose_feats = {f_i: inputs[("color_aug", f_i, 0)] for f_i in self.matching_ids}
            with torch.no_grad():
                for fi in self.matching_ids[1:]:
                    if fi < 0:
                        pose_inputs = [pose_feats[fi], pose_feats[fi + 1]]
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                        axisangle, translation = self.models["pose"](pose_inputs)

                        if self.args.use_velocity:

                            translation_magnitude = translation[:, 0].squeeze(1).norm(p="fro",
                                                                                      dim=1).unsqueeze(1).unsqueeze(2)
                            translation_norm = translation[:, 0] / translation_magnitude
                            translation_norm *= inputs[("displacement_magnitude", f_i)].unsqueeze(1).unsqueeze(2)
                            translation = translation_norm

                        else:
                            translation = translation[:, 0]

                        pose = pose_vec2mat(axisangle[:, 0], translation, invert=True, rotation_mode=self.args.rotation_mode)

                        # now find 0->fi pose
                        if fi != -1:
                            pose = torch.matmul(pose, inputs[('relative_pose', fi + 1)])

                    else:
                        pose_inputs = [pose_feats[fi - 1], pose_feats[fi]]
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                        axisangle, translation = self.models["pose"](pose_inputs)

                        if self.args.use_velocity:

                            translation_magnitude = translation[:, 0].squeeze(1).norm(p="fro",
                                                                                      dim=1).unsqueeze(1).unsqueeze(2)
                            translation_norm = translation[:, 0] / translation_magnitude
                            translation_norm *= inputs[("displacement_magnitude", f_i)].unsqueeze(1).unsqueeze(2)
                            translation = translation_norm

                        else:
                            translation = translation[:, 0]

                        pose = pose_vec2mat(axisangle[:, 0], translation, invert=False, rotation_mode=self.args.rotation_mode)

                        # now find 0->fi pose
                        if fi != 1:
                            pose = torch.matmul(pose, inputs[('relative_pose', fi - 1)])

                    # set missing images to 0 pose
                    for batch_idx, feat in enumerate(pose_feats[fi]):
                        if feat.sum() == 0:
                            pose[batch_idx] *= 0

                    inputs[('relative_pose', fi)] = pose
        else:
            raise NotImplementedError

        return outputs


    def generate_images_pred_for_coarse_model(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """

        for scale in self.args.scales:

            disp = outputs[("disp", scale)]

            disp = F.interpolate(disp, [self.args.height, self.args.width], mode="bilinear", align_corners=False)
            source_scale = 0

            _, depth = disp_to_depth(disp, self.args.min_depth, self.args.max_depth)
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.args.frame_ids[1:]):
                T = outputs[("cam_T_cam", 0, frame_id)]

                cam_points = self.backproject_depth[source_scale](depth, inputs[("theta_lut", frame_id)], inputs[("angle_lut", frame_id)], T)

                pix_coords = self.project_3d[source_scale](cam_points, inputs[("K", frame_id)],
                                           inputs[("D", frame_id)],
                                           self.args.height, self.args.width)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

                if not self.args.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]


    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for ite in range(self.args.iters):
            disp = outputs[("disp", 0, ite)]

            _, depth = disp_to_depth(disp, self.args.min_depth, self.args.max_depth)
            outputs[("depth", 0, ite)] = depth

            for i, frame_id in enumerate(self.args.frame_ids[1:]):
                T = outputs[("cam_T_cam", 0, frame_id)]

                # don't update posenet based on FDG prediction
                T = T.detach()

                source_scale = 0
                scale = 0

                cam_points = self.backproject_depth[source_scale](depth, inputs[("theta_lut", frame_id)], inputs[("angle_lut", frame_id)], T)

                pix_coords = self.project_3d[source_scale](cam_points, inputs[("K", frame_id)],
                                           inputs[("D", frame_id)],
                                           self.args.height, self.args.width)

                outputs[("sample", frame_id, scale, ite)] = pix_coords

                outputs[("color", frame_id, scale, ite)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale, ite)],
                    padding_mode="border", align_corners=True)

                if not self.args.disable_automasking:
                    outputs[("color_identity", frame_id, scale, ite)] = \
                        inputs[("color", frame_id, source_scale)]


    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.args.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss


    @staticmethod
    def compute_loss_masks(reprojection_loss, identity_reprojection_loss):
        """ Compute loss masks for each of standard reprojection and depth hint
        reprojection.

        identity_reprojections_loss and/or depth_hint_reprojection_loss can be None"""

        if identity_reprojection_loss is None:
            # we are not using automasking - standard reprojection loss applied to all pixels
            reprojection_loss_mask = torch.ones_like(reprojection_loss)


        else:
            all_losses = torch.cat([reprojection_loss, identity_reprojection_loss], dim=1)

            idxs = torch.argmin(all_losses, dim=1, keepdim=True)
            reprojection_loss_mask = (idxs != 1).float()  # automask has index '1'

        return reprojection_loss_mask

    def compute_matching_mask(self, outputs):
        """Generate a mask of where we cannot trust the cost volume, based on the difference
        between the cost volume and the teacher, coarse network"""

        coarse_output = outputs[('coarse_depth', 0, 0)]
        matching_depth = 1 / outputs['lowest_cost'].unsqueeze(1).to(self.device)

        # mask where they differ by a large amount
        mask = ((matching_depth - coarse_output) / coarse_output) < 1.0
        mask *= ((coarse_output - matching_depth) / matching_depth) < 1.0
        return mask[:, 0]

    def compute_losses_for_coarse_model(self, inputs, outputs):
        """Compute the reprojection, smoothness and proxy supervised losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.args.scales:
            loss = 0
            reprojection_losses = []

            source_scale = 0

            disp = outputs[("disp", scale)]
            disp = F.interpolate(
                disp, [self.args.height//(2**scale), self.args.width//(2**scale)], mode="bilinear", align_corners=False)

            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.args.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.args.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.args.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                # differently to Monodepth2, compute mins as we go
                identity_reprojection_loss, _ = torch.min(identity_reprojection_losses, dim=1,
                                                              keepdim=True)
            else:
                identity_reprojection_loss = None

            # differently to Monodepth2, compute mins as we go
            reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)

            if not self.args.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).to(self.device) * 0.00001

            # find minimum losses from [reprojection, identity, depth hints reprojection]
            reprojection_loss_mask = self.compute_loss_masks(reprojection_loss, identity_reprojection_loss)

            # standard reprojection loss
            reprojection_loss = reprojection_loss * reprojection_loss_mask
            reprojection_loss = reprojection_loss.sum() / (reprojection_loss_mask.sum() + 1e-7)

            outputs["identity_selection_coarse/{}".format(scale)] = (1 - reprojection_loss_mask).float()
            losses['reproj_loss_coarse/{}'.format(scale)] = reprojection_loss

            loss += reprojection_loss

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.args.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss_coarse/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss_coarse"] = total_loss

        return losses

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection, smoothness and proxy supervised losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for ite in range(self.args.iters):
            loss = 0
            reprojection_losses = []

            i_weight = 0.8 ** (self.args.iters - ite - 1)

            source_scale = 0
            scale = 0

            disp = outputs[("disp", 0, ite)]
            disp = F.interpolate(
                disp, [self.args.height//(2**scale), self.args.width//(2**scale)], mode="bilinear", align_corners=False)

            color = inputs[("color", 0, 0)]
            target = inputs[("color", 0, 0)]

            for frame_id in self.args.frame_ids[1:]:
                pred = outputs[("color", frame_id, 0, ite)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target)*i_weight)
            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.args.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.args.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target)*i_weight)

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                # differently to Monodepth2, compute mins as we go
                identity_reprojection_loss, _ = torch.min(identity_reprojection_losses, dim=1,
                                                              keepdim=True)
            else:
                identity_reprojection_loss = None

            # differently to Monodepth2, compute mins as we go
            reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)

            if not self.args.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).to(self.device) * 0.00001

            reprojection_loss_mask = self.compute_loss_masks(reprojection_loss, identity_reprojection_loss)

            # find which pixels to apply reprojection loss to, and which pixels to apply
            # consistency loss to
            reprojection_loss_mask = torch.ones_like(reprojection_loss_mask)
            if not self.args.disable_motion_masking:
                reprojection_loss_mask = (reprojection_loss_mask *
                                          outputs['consistency_mask'].unsqueeze(1))
            if not self.args.no_matching_augmentation:
                reprojection_loss_mask = (reprojection_loss_mask *
                                          (1 - outputs['augmentation_mask']))
            consistency_mask = (1 - reprojection_loss_mask).float()

            # standard reprojection loss
            reprojection_loss = reprojection_loss * reprojection_loss_mask
            reprojection_loss = reprojection_loss.sum() / (reprojection_loss_mask.sum() + 1e-7)

            # consistency loss:
            # encourage FDG to be like singe frame where masking is happening

            fine_depth = outputs[("depth", 0, ite)]
            # no gradients for mono prediction!
            coarse_depth = outputs[("coarse_depth", 0, scale)].detach()
            consistency_loss = torch.abs(fine_depth - coarse_depth) * consistency_mask
            consistency_loss = consistency_loss.mean()

            # save for logging to tensorboard
            consistency_target = (coarse_depth.detach() * consistency_mask +
                                  fine_depth.detach() * (1 - consistency_mask))
            consistency_target = 1 / consistency_target
            outputs["consistency_target/{}".format(ite)] = consistency_target
            losses['consistency_loss/{}'.format(ite)] = consistency_loss

            losses['reproj_loss/{}'.format(ite)] = reprojection_loss

            loss += reprojection_loss + consistency_loss

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.args.disparity_smoothness * smooth_loss
            total_loss += loss * i_weight
            losses["loss/{}".format(ite)] = loss

        total_loss /= self.args.iters
        losses["loss"] = total_loss

        return losses

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        self.val_iter = iter(self.val_loader)
        inputs = next(self.val_iter)
        self.inputs_to_device(inputs)

        with torch.no_grad():
            outputs, losses = self.predict_depths(inputs)

            self.log("val", inputs, outputs, losses, True)

            del inputs, outputs, losses

        self.set_train()


    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        min_depth = 1e-3
        max_depth = 80

        depth_pred = outputs[("depth", 0, 0)].detach()

        depth_gt = inputs[("depth_gts", 0, 0)][:, None]
        mask = (depth_gt > min_depth) * (depth_gt < max_depth)

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

        self.log_metric(depth_errors)


class DAFANet(DepthEstimationModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.configure_optimizers()
        self.pre_init()
