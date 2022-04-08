# AS-MLP for Semantic Segmentaion

This repo contains the supported code and configuration files to reproduce semantic segmentaion results of [AS-MLP](https://arxiv.org/pdf/2107.08391.pdf). It is based on [Swin Transformer](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation).

## Results and Models

### ADE20K

| Backbone | Method | Crop Size | Lr Schd | mIoU (ms+flip) | #params | FLOPs | config | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AS-MLP-T | UPerNet | 512x512 | 160K | 46.5 | 60M | 937G | [config](configs/asmlp/upernet_asmlp_tiny_patch4_shift5_512x512_160k_ade20k.py) |  
| AS-MLP-S | UperNet | 512x512 | 160K| 49.2 | 81M | 1024G | [config](configs/asmlp/upernet_asmlp_small_patch4_shift5_512x512_160k_ade20k.py) |  
| AS-MLP-B | UperNet | 512x512 | 160K | 49.5 | 121M | 1166G | [config](configs/asmlp/upernet_asmlp_base_patch4_shift5_512x512_160k_ade20k.py) | [onedrive](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/liandz_shanghaitech_edu_cn/Eee6yxLZ27hAkk3yVAD6UtsB189avw7EFHN0cy59L9LOeA?e=Zc1ro0) |

**Notes**: 

- **Pre-trained models can be downloaded from [AS-MLP for ImageNet Classification](https://github.com/svip-lab/AS-MLP)**.


## Results of MoBY with Swin Transformer


## Usage

### Installation

Please refer to [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/get_started.md#installation) for installation and dataset preparation.

### Inference
```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <SEG_CHECKPOINT_FILE> --eval mIoU

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <SEG_CHECKPOINT_FILE> <GPU_NUM> --eval mIoU

# multi-gpu, multi-scale testing
tools/dist_test.sh <CONFIG_FILE> <SEG_CHECKPOINT_FILE> <GPU_NUM> --aug-test --eval mIoU
```

### Training

To train with pre-trained models, run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE> --options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments] 
```
For example, to train an UPerNet model with a `AS-MLP-T` backbone and 8 gpus, run:
```
tools/dist_train.sh configs/asmlp/upernet_asmlp_tiny_patch4_shift5_512x512_160k_ade20k.py 8 --options model.pretrained=<PRETRAIN_MODEL> 
```

**Notes:** 
- `use_checkpoint` is used to save GPU memory. Please refer to [this page](https://pytorch.org/docs/stable/checkpoint.html) for more details.
- The default learning rate and training schedule is for 8 GPUs and 2 imgs/gpu.


## Citation
```
@article{Lian_2021_ASMLP,
  author = {Lian, Dongze and Yu, Zehao and Sun, Xing and Gao, Shenghua},
  title = {AS-MLP: An Axial Shifted MLP Architecture for Vision},
  journal={ICLR},
  year = {2022}
}
```

## Other Links

> **Image Classification**: See [AS-MLP for Image Classification](https://github.com/svip-lab/AS-MLP).

> **Object Detection**: See [AS-MLP for Object Detection](https://github.com/svip-lab/AS-MLP-Object-Detection).
