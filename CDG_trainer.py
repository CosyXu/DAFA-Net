"""
Depth Estimation training for CDG.

Parts of the code adapted from https://github.com/valeoai/WoodScape.
Please refer to the license of the above repo.

"""

import time

import torch
from colorama import Fore, Style
from torch.utils.data import DataLoader

from data_loader.woodscape_loader import WoodScapeRawDataset
from data_loader.synwoodscape_loader import SynWoodScapeRawDataset
from CDG_loss import PhotometricFisheye, PhotometricReconstructionLoss

from utils import tensor2array, pose_vec2mat
from train_utils import TrainUtils


from models.CDG.depth_decoder import ResDecoder, SwinDecoder
from models.CDG.pose_decoder import PoseDecoder
from models.CDG.resnet_encoder import ResnetEncoder
from models.CDG.swin_encoder import get_orgwintrans_backbone


class CDGModelBase(TrainUtils):
    def __init__(self, args):
        super().__init__(args)

        # --- INIT MODELS ---
        if 'orgSwin' in self.args.encoder_model_type:
            self.models["coarse_encoder"] = get_orgwintrans_backbone(self.args.swin_model_type, pretrained=True).to(self.device)
            self.models["coarse_depth"] = SwinDecoder(num_ch_enc=self.models["coarse_encoder"].num_ch_enc).to(self.device)
        elif 'Res' in self.args.encoder_model_type:
            self.models["coarse_encoder"] = ResnetEncoder(num_layers=self.args.network_layers, pretrained=True).to(self.device)
            self.models["coarse_depth"] = ResDecoder(num_ch_enc=self.models["coarse_encoder"].num_ch_enc).to(self.device)
        else:
            raise NotImplementedError

        self.parameters_to_train += list(self.models["coarse_encoder"].parameters())
        self.parameters_to_train += list(self.models["coarse_depth"].parameters())

        # --- Init Pose model ---
        self.models["pose_encoder"] = ResnetEncoder(num_layers=self.args.pose_network_layers,
                                                    pretrained=True,
                                                    num_input_images=2).to(self.device)

        self.models["pose"] = PoseDecoder(self.models["pose_encoder"].num_ch_enc,
                                          num_input_features=1,
                                          num_frames_to_predict_for=2).to(self.device)

        self.parameters_to_train += list(self.models["pose_encoder"].parameters())
        self.parameters_to_train += list(self.models["pose"].parameters())

        print(f"{Fore.BLUE}=> Training on the {args.dataset.upper()} projection model \n"
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

        inverse_warp = PhotometricFisheye(args)
        self.photometric_losses = PhotometricReconstructionLoss(inverse_warp, args)

    def pre_init(self):
        if self.args.pretrained_weights:
            self.load_model()

        if 'cuda' in self.device:
            torch.cuda.synchronize()

    def depth_train(self):
        """Trainer function for depth prediction on fisheye images"""

        for self.epoch in range(self.args.epochs):
            # switch to train mode
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
                outputs = self.predict_depths(inputs)

                # -- POSE ESTIMATION --
                outputs.update(self.predict_poses(inputs))

                # -- PHOTOMETRIC LOSSES --
                losses, outputs = self.photometric_losses(inputs, outputs)

                # -- COMPUTE GRADIENT AND DO OPTIMIZER STEP --
                self.optimizer.zero_grad()
                losses["depth_loss"].mean().backward()
                self.optimizer.step()

                duration = time.time() - before_op_time
                gpu_time += duration

                if batch_idx % self.args.log_frequency == 0:
                    self.log_time(batch_idx, duration, losses["depth_loss"].mean().cpu().data,
                                  data_loading_time, gpu_time)
                    self.depth_statistics("train", inputs, outputs, losses)
                    data_loading_time = 0
                    gpu_time = 0

                    self.val()


                self.step += 1
                before_op_time = time.time()


            self.lr_scheduler.step()

            if (self.epoch + 1) % self.args.save_frequency == 0 and self.epoch > 15:
                self.save_model()

        # self.save_model()
        print("Training finished!")

    def predict_depths(self, inputs, features=None):
        """Predict depths for target frame or for all monocular sequences."""
        outputs = dict()
        features = self.models["coarse_encoder"](inputs["color_aug", 0, 0])
            
        outputs.update(self.models["coarse_depth"](features))

        return outputs

    def predict_poses(self, inputs):
        """Predict poses between input frames for monocular sequences."""
        outputs = dict()
        # Compute the pose to each source frame via a separate forward pass through the pose network.
        # select what features the pose network takes as input
        pose_feats = {frame_id: inputs[("color_aug", frame_id, 0)] for frame_id in self.args.frame_ids}

        for frame_id in self.args.frame_ids[1:]:
            # To maintain ordering we always pass frames in temporal order
            if frame_id == -1:
                pose_inputs = [pose_feats[frame_id], pose_feats[0]]
            else:
                pose_inputs = [pose_feats[0], pose_feats[frame_id]]

            pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                
            axisangle, translation = self.models["pose"](pose_inputs)    # [2,2,1,3]  [2,2,1,3]

            # Normalize the translation vec and multiply by the displacement magnitude obtained from speed
            # of the vehicle to scale it to the real world translation
            translation_magnitude = translation[:, 0].squeeze(1).norm(p="fro",
                                                                      dim=1).unsqueeze(1).unsqueeze(2)
            translation_norm = translation[:, 0] / translation_magnitude
            translation_norm *= inputs[("displacement_magnitude", frame_id)].unsqueeze(1).unsqueeze(2)
            translation = translation_norm

            outputs[("axisangle", 0, frame_id)] = axisangle
            outputs[("translation", 0, frame_id)] = translation
            # Invert the matrix if the frame id is negative
            outputs[("cam_T_cam", 0, frame_id)] = pose_vec2mat(axisangle[:, 0],
                                                               translation,
                                                               invert=(frame_id < 0),
                                                               rotation_mode=self.args.rotation_mode)
        return outputs

    def depth_statistics(self, mode, inputs, outputs, losses) -> None:
        """Print the weights and images"""
        writer = self.writers[mode]
        for loss, value in losses.items():
            writer.add_scalar(f"{loss}", value.mean(), self.step)
        writer.add_scalar("learning_rate", self.optimizer.param_groups[0]['lr'], self.step)

        for j in range(min(4, self.args.batch_size)):  # write maxmimum of four images
            for s in range(self.args.num_scales):
                for frame_id in self.args.frame_ids:
                    writer.add_image(f"color_{frame_id}/{j}",
                                     inputs[("color", frame_id, 0)][j].data, self.step)
                    if s == 0 and frame_id == 0:
                        writer.add_image(f"disp_{frame_id}_{s}/{j}",
                                         tensor2array(outputs[("disp", s)][j, 0], colormap='magma'), self.step)
                if s == 0:
                    writer.add_image(f"color_pred_-1_{s}/{j}",
                                     outputs[("color", -1, s)][j].data, self.step)


    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()

        self.val_iter = iter(self.val_loader)
        inputs = next(self.val_iter)
        self.inputs_to_device(inputs)

        with torch.no_grad():
            # -- DEPTH ESTIMATION --
            outputs = self.predict_depths(inputs)

            # -- POSE ESTIMATION --
            outputs.update(self.predict_poses(inputs))

            # -- PHOTOMETRIC LOSSES --
            losses, outputs = self.photometric_losses(inputs, outputs)
            self.depth_statistics("val", inputs, outputs, losses)

            del inputs, outputs, losses

        self.set_train()


class CDGModel(CDGModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.configure_optimizers()
        self.pre_init()
