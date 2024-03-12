"""
Evaluation script for DAFA-Net.

"""

import os
from tqdm import tqdm

import torch
import torch.nn.functional as F

import yaml

from main import collect_args

from torch.utils.data import DataLoader
from data_loader.synwoodscape_loader import SynWoodScapeRawDataset
from train_utils import Tupperware

from metric import Metric

from layers import disp_to_depth

from models.CDG.depth_decoder import ResDecoder, SwinDecoder
from models.CDG.pose_decoder import PoseDecoder
from models.CDG.resnet_encoder import ResnetEncoder
from models.CDG.swin_encoder import get_orgwintrans_backbone
from models.FDG.hr_encoder import hrnet18
from models.FDG.enhancement_encoder import EnhancementEncoder
from models.FDG.recurrent_decoder import RecurrentDecoder

FRAME_RATE = 1

def inputs_to_device(inputs, device):
    for key, ipt in inputs.items():
        inputs[key] = ipt.to(device)


@torch.no_grad()
def test_simple(args):
    """Function to predict for a single image or folder of images"""

    if not os.path.isdir(args.output_directory):
        os.mkdir(args.output_directory)

    frames_to_load = [0]
    if args.use_future_frame:
        frames_to_load.append(1)
    for idx in range(-1, -1 - args.num_matching_frames, -1):
        if idx not in frames_to_load:
            frames_to_load.append(idx)


    coarse_encoder_path = os.path.join(args.load_weights_folder, "coarse_encoder.pth")
    coarse_depth_path = os.path.join(args.load_weights_folder, "coarse_depth.pth")

    coarse_encoder_dict = torch.load(coarse_encoder_path)
    coarse_depth_dict = torch.load(coarse_depth_path)

    if 'orgSwin' in args.encoder_model_type:
        coarse_encoder = get_orgwintrans_backbone(backbone_name=args.swin_model_type, pretrained=False).to(args.device)
        coarse_depth = SwinDecoder(num_ch_enc=coarse_encoder.num_ch_enc).to(args.device)
    elif 'Res' in args.encoder_model_type:
        coarse_encoder = ResnetEncoder(num_layers=args.network_layers, pretrained=False).to(args.device)
        coarse_depth = ResDecoder(num_ch_enc=coarse_encoder.num_ch_enc).to(args.device)


    coarse_encoder.load_state_dict(coarse_encoder_dict)
    coarse_depth.load_state_dict(coarse_depth_dict)

    coarse_encoder.eval()
    coarse_depth.eval()

    encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
    depth_decoder_path = os.path.join(args.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)
    depth_decoder_dict = torch.load(depth_decoder_path)

    encoder = EnhancementEncoder(
        pretrained=False,
        input_width=encoder_dict['width'],
        input_height=encoder_dict['height'],
        depth_binning=args.depth_binning,
        num_depth_bins=args.num_depth_bins,
        cost_volume_mode=args.cost_volume_mode,
        device=args.device).to(args.device)

    depth_decoder = RecurrentDecoder(num_ch_enc=encoder.num_ch_enc,
                                                           iters=args.iters).to(args.device)


    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(depth_decoder_dict)

    encoder.eval()
    depth_decoder.eval()

    encoder_context = hrnet18(False).to(args.device)
    encoder_context_path = os.path.join(args.load_weights_folder, "encoder_context.pth")
    encoder_context.load_state_dict(torch.load(encoder_context_path))
    encoder_context.eval()


    pose_enc_dict = torch.load(os.path.join(args.load_weights_folder, "pose_encoder.pth"))
    pose_dec_dict = torch.load(os.path.join(args.load_weights_folder, "pose.pth"))

    pose_enc = ResnetEncoder(args.pose_network_layers, False, num_input_images=2).to(args.device)
    pose_dec = PoseDecoder(pose_enc.num_ch_enc, num_input_features=1,
                           num_frames_to_predict_for=2).to(args.device)

    pose_enc.load_state_dict(pose_enc_dict, strict=True)
    pose_dec.load_state_dict(pose_dec_dict, strict=True)

    pose_enc.eval()
    pose_dec.eval()

    depth_bin_facs = encoder_dict.get('depth_bin_facs').to(args.device)

    try:
        HEIGHT, WIDTH = encoder_dict['height'], encoder_dict['width']
    except KeyError:
        print('No "height" or "width" keys found in the encoder state_dict')
        HEIGHT, WIDTH = args.height, args.width

    print("-> Computing predictions with size {}x{}".format(HEIGHT, WIDTH))

    val_dataset = SynWoodScapeRawDataset(data_path=args.dataset_dir,
                                         path_file=args.val_file,
                                         is_train=False,
                                         config=args)

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=True)

    metric_func = Metric(args.metric_name, None)

    print(f"=> Predicting on {len(val_dataset)} validation images")

    # do inference
    with torch.no_grad():
        for i, data in tqdm(enumerate(val_loader)):
            inputs_to_device(data, args.device)

            input_color = data[('color', 0, 0)]

            coarse_feats = coarse_encoder(input_color)
            coarse_output = coarse_depth(coarse_feats)

            _, depth_prior = disp_to_depth(coarse_output[("disp", 0)], args.min_depth, args.max_depth)

            encoder_output = encoder.forward_test(input_color,
                                                  F.interpolate(depth_prior,
                                                                [HEIGHT // 4, WIDTH // 4],
                                                                mode="bilinear"),
                                                  depth_bin_facs)

            context_output = encoder_context(input_color)

            output, _, _ = depth_decoder(encoder_output, context_output, test_mode=True)

            pred_disp, _ = disp_to_depth(output, args.min_depth, args.max_depth)

            scaled_dist = 1 / pred_disp[:, 0]

            depth_gt = data[("depth_gts", 0, 0)]
            metric_func.update_metric(scaled_dist, depth_gt)

    info_line, err_line = metric_func.get_metric_output(test_mode=True)

    print(info_line)
    print(err_line)



if __name__ == '__main__':
    config = collect_args()
    params = yaml.safe_load(open(config.config))
    args = Tupperware(params)
    
    test_simple(args)


