"""
Utilities for DAFA-Net.

Parts of the code adapted from https://github.com/nianticlabs/monodepth2
Please refer to the license of the above repo.

"""

import os
import time

import numpy as np
import torch
from colorama import Fore, Style
from ruamel import yaml
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import matplotlib.pyplot as plt


class TrainUtils:
    def __init__(self, args):
        """Train Utils class providing training utilities for depth semantic and motion estimation
        :param args: input params from config file
        """
        self.args = args
        self.device = args.device
        self.log_path = os.path.join(args.output_directory, args.model_name)
        assert args.height % 32 == 0, "'height' must be a multiple of 32"
        assert args.width % 32 == 0, "'width' must be a multiple of 32"
        self.models = dict()
        self.parameters_to_train = []
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        self.trans_pil = transforms.ToPILImage()
        self.optimizer = None
        self.lr_scheduler = None

        self.num_scales = len(args.scales)
        self.num_input_frames = len(args.frame_ids)
        self.num_pose_frames = 2
        self.train_teacher_and_pose = not self.args.freeze_teacher_and_pose

        assert self.args.frame_ids[0] == 0, "frame_ids must start with 0"
        assert len(self.args.frame_ids) > 1, "frame_ids must have more than 1 frame specified"

        self.writers = dict()
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

    def inputs_to_device(self, inputs):
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

    def set_train(self):
        """Convert all models to training mode"""
        for k, m in self.models.items():
            if self.train_teacher_and_pose:
                m.train()
            else:
                # if teacher + pose is frozen, then only use training batch norm stats for
                # multi components
                if k in ['depth', 'encoder', 'encoder_context']:
                    m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode"""
        for m in self.models.values():
            m.eval()


    def log_time(self, batch_idx, duration, loss, data_time, gpu_time):
        """Print a logging statement to the terminal"""
        samples_per_sec = self.args.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print(f"{Fore.GREEN}epoch {self.epoch:>3}{Style.RESET_ALL} "
              f"| batch {batch_idx:>6} "
              f"| current lr {self.optimizer.param_groups[0]['lr']:.4f} "
              f"| examples/s: {samples_per_sec:5.1f} "
              f"| {Fore.RED}loss: {loss:.5f}{Style.RESET_ALL} "
              f"| {Fore.BLUE}time elapsed: {self.sec_to_hm_str(time_sofar)}{Style.RESET_ALL} "
              f"| {Fore.CYAN}time left: {self.sec_to_hm_str(training_time_left)}{Style.RESET_ALL} "
              f"| CPU/GPU time: {data_time:0.1f}s/{gpu_time:0.1f}s")

    def log_metric(self, depth_errors):
        for i, metric in enumerate(self.depth_metric_names):
            print(f"{Fore.CYAN}{metric}: {np.array(depth_errors[i].cpu()):.3f}{Style.RESET_ALL} |", end=' ')
        print()

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters_to_train, self.args.learning_rate)
        # self.optimizer = torch.optim.NAdam(self.parameters_to_train, self.args.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.args.scheduler_step_size, 0.1)


    def save_model(self):
        """Save model weights to disk"""
        save_folder = os.path.join(self.log_path, "models", f"weights_{self.epoch}", str(self.step))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, f"{model_name}.pth")
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.args.height
                to_save['width'] = self.args.width
                to_save['depth_bin_facs'] = self.depth_bin_facs
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "adam.pth")
        if self.epoch > 50:  # Optimizer file is quite large! Sometimes, life is a compromise.
            torch.save(self.optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk"""
        self.args.pretrained_weights = os.path.expanduser(self.args.pretrained_weights)

        assert os.path.isdir(self.args.pretrained_weights), f"Cannot find folder {self.args.pretrained_weights}"
        print(f"=> Loading model from folder {self.args.pretrained_weights}")

        for n in self.args.models_to_load:
            print(f"Loading {n} weights...")
            path = os.path.join(self.args.pretrained_weights, f"{n}.pth")
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path, map_location=self.args.device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading optimizer state
        if not self.args.freeze_encoder:
            optimizer_load_path = os.path.join(self.args.pretrained_weights, f"{self.args.optimizer}.pth")
            if os.path.isfile(optimizer_load_path):
                print(f"Loading {self.args.optimizer} weights")
                optimizer_dict = torch.load(optimizer_load_path, map_location=self.args.device)
                self.optimizer.load_state_dict(optimizer_dict)
            else:
                print(f"Cannot find {self.args.optimizer} weights so {self.args.optimizer} is randomly initialized")

    def sec_to_hm(self, t):
        """Convert time in seconds to time in hours, minutes and seconds
        e.g. 10239 -> (2, 50, 39)
        """
        t = int(t)
        s = t % 60
        t //= 60
        m = t % 60
        t //= 60
        return t, m, s

    def sec_to_hm_str(self, t):
        """Convert time in seconds to a nice string
        e.g. 10239 -> '02h50m39s'
        """
        h, m, s = self.sec_to_hm(t)
        return f"{h:02d}h{m:02d}m{s:02d}s"


    def log(self, mode, inputs, outputs, losses, if_gru=False):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.args.batch_size)):  # write a maxmimum of four images
            s = 0  # log only max scale
            ite = 0
            for frame_id in self.args.frame_ids:
                writer.add_image(
                    "color_{}_{}/{}".format(frame_id, s, j),
                    inputs[("color", frame_id, s)][j].data, self.step)
                if s == 0 and frame_id != 0:
                    if if_gru:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s, ite)][j].data, self.step)
                    else:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)
            if if_gru:
                disp = colormap(outputs[("disp", s, ite)][j, 0])
            else:
                disp = colormap(outputs[("disp", s)][j, 0])

            writer.add_image(
                "disp_multi_{}/{}".format(s, j),
                disp, self.step)

            disp = colormap(outputs[('coarse_disp', s)][j, 0])
            writer.add_image(
                "disp_coarse/{}".format(j),
                disp, self.step)

            if outputs.get("lowest_cost") is not None:
                lowest_cost = outputs["lowest_cost"][j]

                consistency_mask = \
                    outputs['consistency_mask'][j].cpu().detach().unsqueeze(0).numpy()

                min_val = np.percentile(lowest_cost.numpy(), 10)
                max_val = np.percentile(lowest_cost.numpy(), 90)
                lowest_cost = torch.clamp(lowest_cost, min_val, max_val)
                lowest_cost = colormap(lowest_cost)

                writer.add_image(
                    "lowest_cost/{}".format(j),
                    lowest_cost, self.step)
                writer.add_image(
                    "lowest_cost_masked/{}".format(j),
                    lowest_cost * consistency_mask, self.step)
                writer.add_image(
                    "consistency_mask/{}".format(j),
                    consistency_mask, self.step)

                consistency_target = colormap(outputs["consistency_target/0"][j])
                writer.add_images(
                    "consistency_target/{}".format(j),
                    consistency_target, self.step)


_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)  # for plotting

def colormap(inputs, normalize=True, torch_transpose=True):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()

    vis = inputs
    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d

    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis

class Tupperware(dict):
    MARKER = object()

    def __init__(self, value=None):
        if value is None:
            pass
        elif isinstance(value, dict):
            for key in value:
                self.__setitem__(key, value[key])
        else:
            raise TypeError('expected dict')

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, Tupperware):
            value = Tupperware(value)
        super(Tupperware, self).__setitem__(key, value)

    def __getitem__(self, key):
        found = self.get(key, Tupperware.MARKER)
        if found is Tupperware.MARKER:
            found = Tupperware()
            super(Tupperware, self).__setitem__(key, found)
        return found

    __setattr__, __getattr__ = __setitem__, __getitem__
