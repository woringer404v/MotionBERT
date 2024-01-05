# Semester Project - VITA LAB:  Transformers with 3D heatmaps for skeletal action recognition.

This repository uses the following models:

[Pyskl](https://github.com/kennymckormick/pyskl)
[MotionBERT](https://github.com/Walter0807/MotionBERT)
[ViViT](https://github.com/google-research/scenic/tree/main/scenic/projects/vivit) which is the official code of the paper but since it is in JAX, another adaptation is presented here using TensorFlow 

## Installation

```bash
conda create -n motionbert python=3.7 anaconda
conda activate motionbert
# Please install PyTorch according to your CUDA version.
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Dataset 

We use the pre-processed 2D and 3D skeleton annotations provided by [Pyskl](https://github.com/kennymckormick/pyskl):
- NTURGB+D [2D Skeleton]: https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_hrnet.pkl
- NTURGB+D [3D Skeleton]: https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_3danno.pkl

Please place both files in folder data/action

## Training

### MotionBERT

```bash
python train_action_mbert.py --config configs/action/MB_train.yaml --checkpoint checkpoint/action/mbert
```

### ViViT

```bash
python train_action_vivit.py --checkpoint checkpoint/action/vivit
```

### Vanilla transformer

```bash
python train_action_vanilla.py  --checkpoint checkpoint/action/vanilla
```



## Model Performance Summary

| Model                          | 3D Heatmap Performance | Number of Parameters in millions | Checkpoint Link   |
|--------------------------------|------------------------|----------------------|-------------------|
| MotionBERT                     | 87.8\%                 | 63.5M                | [Link](X)         |
| MotionBERT + Augmentation      | 87.2\%                 | 63.5M                | [Link](X)         |
| ViViT (Model 1)                | 81.3\%                 | 8.65M                | [Link](X)         |
| Big ViViT (Model 1)            | 79.6\%                 | 63.7M                | [Link](X)         |
| Small Tub ViViT (Model 1)      | 82.3\%                 | 7.6M                 | [Link](X)         |
| ViViT (Model 2)                | 80.9\%                 | 44.4M                | [Link](X)         |
| Vanilla Transformer            | 82.2\%                 | 9.5M                 | [Link](X)         |
| Big Vanilla Transformer        | 83.4\%                 | 94.7M                | [Link](X)         |
| Vanilla Transformer + Aug      | 84.3\%                 | 94.7M                | [Link](X)         |



