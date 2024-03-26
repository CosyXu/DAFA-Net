# DAFA-Net


> **Distortion-Aware Feature Aggregation for Self-Supervised Fisheye Monocular Depth Estimation** 

#### 1. Install

Experiments were conducted with PyTorch 1.8.0, CUDA 11.1, Python 3.7, and Ubuntu 20.04. Other configurations meeting these requirements should also work. Use Anaconda to create a conda environment:

```shell
conda create -n DAFA python=3.7
conda activate DAFA
```

Install PyTorch and dependencies:

```shell
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install colorama==0.4.6 matplotlib==3.5.3 numpy==1.21.5 Pillow==9.3.0 PyYAML==6.0.1 timm==0.6.13 tqdm==4.64.1 yacs==0.1.6 
pip install ruamel.yaml
pip install future tensorboard
```

#### 2. Prepare datasets

```shell
git clone -b dev https://github.com/CosyXu/DAFA-Net.git
cd DAFA-Net
mkdir datasets
mkdir outputs
```

Follow the instructions on [OmniDet](https://github.com/valeoai/WoodScape/blob/master/README.md) to prepare the WoodScape dataset and place it in the `datasets` directory.

```shell
WoodScape    
│
└───rgb_images
│   │   00000_[CAM].png
│   │   00001_[CAM].png
|   |   ...
│   │
└───previous_images
│   │   00000_[CAM]_prev.png
│   │   00001_[CAM]_prev.png
|   |   ...
│   │
└───vehicle_data
│   │
│   └───rgb_images
│   │   00000_[CAM].json
│   │   00001_[CAM].json
|   |   ...
│   │
│   └───previous_images
│   │   00000_[CAM].json
│   │   00001_[CAM].json
|   |   ...
│   │
└───calibration_data
│   │
│   └───calibration
│   │   00000_[CAM].json
│   │   00001_[CAM].json
|   |   ...
│   │
```

[CAM] : FV, RV, MVL, MVR

You can also prepare the SynWoodScape dataset from [here](https://drive.google.com/drive/folders/1N5rrySiw1uh9kLeBuOblMbXJ09YsqO7I). The data organization is similar to OmniDet's. Place it in the `datasets` directory.

```shell
SynWoodScape    
│
└───rgb_images
│   │   00000_[CAM].png
│   │   00001_[CAM].png
|   |   ...
│   │
└───previous_images
│   │   00000_[CAM]_prev.png
│   │   00001_[CAM]_prev.png
|   |   ...
│   │
└───vehicle_data
│   │
│   └───rgb_images
│   │   00000.txt
│   │   00001.txt
|   |   ...
│   │
│   └───previous_images
│   │   00000.txt
│   │   00001.txt
|   |   ...
│   │
└───calibration_data
│   │   [CAM].json
│   │   [CAM].json
|   |   ...
│   │
└───depth_maps
│   │
│   └───raw_data
│   │   00000_[CAM].json
│   │   00001_[CAM].json
|   |   ...
│   │
```

[CAM] : FV, RV, MVL, MVR

#### 3. Get pre-trained models

You can download the pretrained CDG and DAFA-Net from [here](https://drive.google.com/drive/folders/1F8p9jKlfrTrPCcOMx7JA0NlTcCXM5Kc6?usp=sharing).

| Model Name | Abs Rel. | Sq Rel. | RMSE  | RMSElog | A1    | A2    | A3    |
| ---------- | -------- | ------- | ----- | ------- | ----- | ----- | ----- |
| CDG        | 0.213    | 1.003   | 5.093 | 0.275   | 0.643 | 0.919 | 0.976 |
| DAFA-Net   | 0.203    | 0.881   | 4.836 | 0.262   | 0.663 | 0.927 | 0.979 |

#### 4. Train the Coarse Depth Generator
First, download the pretrained Swin Transformer (Tiny Size) from their [official repository](https://github.com/microsoft/Swin-Transformer). Remember to set the path in `models/CDG/swin_encoder.py` to the downloaded model. Then set `training_mode` to `CDG`, and configure `dataset_dir` and `dataset` to the corresponding configs for WoodScape in `data/params.yaml` to start training:

```shell
python main.py
```

#### 5. Train the DAFA-Net

First, set the path to the weights obtained from training CDG in the `pretrained_weights` field in `data/params.yaml`. Then set `training_mode` to `DAFA-Net`, and configure `dataset_dir` and `dataset` to the corresponding configs for SynWoodScape in `data/params.yaml` to begin training:

```shell
python main.py
```

#### 6. Evaluate the DAFA-Net

Set the path to the trained DAFA-Net weights in the `load_weights_folder` field and set the `batch_size` to 1 in `data/params.yaml` to start the evaluation:
```shell
python test_model.py
```
#### Acknowledgment
Parts of this repository are derived from [Monodepth2](https://github.com/nianticlabs/monodepth2), [OmniDet](https://github.com/valeoai/WoodScape), [ManyDepth](https://github.com/nianticlabs/manydepth), and [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation). We would like to express our gratitude to the authors of these works.
