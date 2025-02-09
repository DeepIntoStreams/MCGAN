# MCGAN for StyleGAN2 

This repo is implemented upon [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) with minimal modifications to implement MCGAN algorithm in PyTorch. Please check the [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) README for the dependencies and the other usages of this codebase.


## Train MCGAN
### CIFAR-10
To train an MCGAN on CIFAR-10 dataset
Download the [CIFAR-10 python version](https://www.cs.toronto.edu/~kriz/cifar.html) and convert to ZIP archive:

```.bash
python dataset_tool.py --source=~/downloads/cifar-10-python.tar.gz --dest=~/datasets/cifar10.zip
```

then run the following line:
```bash
python train.py --outdir=training-runs/MCGAN --data=datasets/cifar10.zip --gpus=4 --mc_size=4 
```

### FFHQ256
Download FFHQ256 dataset in [ffhq-dataset](https://github.com/NVlabs/ffhq-dataset
), and put it under `./data`. Use the following example to scale down the resolution:
```bash
python dataset_tool.py --source=/tmp/images1024x1024 --dest=~/datasets/ffhq-256x256.zip \
    --resolution=256x256
```
Then run the following line for a 4-gpu experiment:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --outdir=training-runs/ffhq256 --data=data/ffhq256x256.zip --cfg=stylegan2 --gpus=4 --batch=32 --gamma=1 --mirror=1 --glr=0.0025 --dlr=0.0025 --cbase=16384 --mc_size 1
```

### Imagenet64
Download Imagenet64 dataset from [Image-net](https://image-net.org/download-images.php), and put it under `./data`. Then run the following line for a 4-gpu experiment:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --outdir=training-runs/imagenet64 --data=data/imagenet64.zip --cfg=stylegan2 --gpus=4 --batch=64 --gamma=0.0128 --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=16384 --cond=1 --snap=100 --mc_size 1
```

### LSUN bedroom 256:
Download FFHQ256 dataset by following instructions in [LSUN](https://github.com/fyu/lsun), and put it under `./data`. Then run the following line for a 4-gpu experiment:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --outdir=training-runs/lsunbed --data=data/lsun_bedroom_256 --cfg=stylegan2 --gpus=4 --batch=32 --gamma=10 --mirror=1 --glr=0.0025 --dlr=0.0025 --cbase=16384 --mc_size 1
```








