"""
Launch script for DAFA-Net.

Parts of the code adapted from https://github.com/valeoai/WoodScape.
Please refer to the license of the above repo.

"""

import argparse
import os
import shutil
from distutils.util import strtobool
from pathlib import Path

import yaml

from train_utils import Tupperware


def collect_args() -> argparse.Namespace:
    """Set command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Config file", type=str, default=Path(__file__).parent / "data/params.yaml")
    args = parser.parse_args()
    return args


def collect_tupperware() -> Tupperware:
    config = collect_args()
    params = yaml.safe_load(open(config.config))
    args = Tupperware(params)
    return args


def main():
    args = collect_tupperware()
    log_path = os.path.join(args.output_directory, args.model_name)

    if os.path.isdir(log_path):
        # pass
        if strtobool(input("=> Clean up the log directory?")):
            shutil.rmtree(log_path, ignore_errors=False, onerror=None)
            os.mkdir(log_path)
            print("=> Cleaned up the logs!")
        else:
            print("=> No clean up performed!")
    else:
        print(f"=> No pre-existing directories found for this experiment. \n"
              f"=> Creating a new one!")
        os.mkdir(log_path)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices or "-1"

    if args.training_mode == "CDG":
        model = CDGModel(args)
        model.depth_train()
    elif args.training_mode == "DAFA-Net":
        model = DAFANet(args)
        model.depth_train()



if __name__ == "__main__":

    from FDG_trainer import DAFANet
    from CDG_trainer import CDGModel

    main()
